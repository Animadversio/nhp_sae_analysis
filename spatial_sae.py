"""
spatial_sae.py
==============
Self-contained Spatial SAE using MatryoshkaBatchTopKSAE trained with a
spatial contrastive loss on adjacent DINOv2 patch pairs.

Public API
----------
    train_spatial_sae(patch_tokens, n_steps, dict_size, k, device, ...)
        -> MatryoshkaBatchTopKSAE

    encode_patches(ae, patch_tokens, device, batch_size)
        -> np.ndarray  shape (N, 196, dict_size)

    mean_pool_codes(codes)
        -> np.ndarray  shape (N, dict_size)

All source code from temporal-saes and vision_patch_pairs is inlined
so this file has no local imports beyond standard PyPI packages.

Dependencies: torch, numpy, einops
"""

from __future__ import annotations

import math
import random
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility functions (from temporal-saes/trainers/trainer.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _set_decoder_norm_to_unit_norm(W_dec_DF: torch.Tensor,
                                    activation_dim: int, d_sae: int) -> torch.Tensor:
    D, F = W_dec_DF.shape
    assert D == activation_dim and F == d_sae
    eps = torch.finfo(W_dec_DF.dtype).eps
    norm = torch.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


@torch.no_grad()
def _remove_gradient_parallel_to_decoder_directions(
        W_dec_DF: torch.Tensor, W_dec_DF_grad: torch.Tensor,
        activation_dim: int, d_sae: int) -> torch.Tensor:
    D, F = W_dec_DF.shape
    assert D == activation_dim and F == d_sae
    normed = W_dec_DF / (torch.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)
    parallel = einops.einsum(W_dec_DF_grad, normed, "d f, d f -> f")
    W_dec_DF_grad -= einops.einsum(parallel, normed, "f, d f -> d f")
    return W_dec_DF_grad


# ---------------------------------------------------------------------------
# MatryoshkaBatchTopKSAE  (from temporal-saes/trainers/matryoshka_batch_top_k.py)
# ---------------------------------------------------------------------------

class MatryoshkaBatchTopKSAE(nn.Module):
    """
    Matryoshka Batch Top-K Sparse Autoencoder.
    Source: temporal-saes/dictionary_learning/dictionary_learning/trainers/matryoshka_batch_top_k.py
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int,
                 group_sizes: list[int]):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert sum(group_sizes) == dict_size
        assert all(s > 0 for s in group_sizes)
        assert isinstance(k, int) and k > 0

        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))

        self.active_groups = len(group_sizes)
        group_indices = [0] + list(torch.cumsum(torch.tensor(group_sizes), dim=0))
        self.group_indices = group_indices
        self.register_buffer("group_sizes", torch.tensor(group_sizes))

        self.W_enc = nn.Parameter(torch.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(dict_size, activation_dim))
        )
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        self.W_dec.data = _set_decoder_norm_to_unit_norm(
            self.W_dec.data.T, activation_dim, dict_size
        ).T
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(self, x: torch.Tensor, return_active: bool = False,
               use_threshold: bool = True) -> torch.Tensor:
        post_relu = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if use_threshold:
            f = post_relu * (post_relu > self.threshold)
        else:
            flat = post_relu.flatten()
            topk = flat.topk(self.k * x.size(0), sorted=False)
            f = (torch.zeros_like(flat)
                 .scatter_(-1, topk.indices, topk.values)
                 .reshape(post_relu.shape))

        max_idx = self.group_indices[self.active_groups]
        f[:, max_idx:] = 0

        if return_active:
            return f, f.sum(0) > 0, post_relu
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        f = self.encode(x)
        x_hat = self.decode(f)
        return (x_hat, f) if output_features else x_hat


# ---------------------------------------------------------------------------
# Matryoshka cumulative reconstruction loss (from temporal-saes design)
# ---------------------------------------------------------------------------

def _matryoshka_recon_loss(
    ae: "MatryoshkaBatchTopKSAE",
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Cumulative Matryoshka reconstruction loss — exact temporal-saes design.

    From MatryoshkaBatchTopKTrainer.loss() in AI4LIFE-GROUP/temporal-saes:

      1. Encode ONCE with full-dict batch top-k (use_threshold=False).
      2. Split features and W_dec into per-group chunks.
      3. Cumulatively add each group's contribution to the reconstruction.
      4. Compute sum-of-squares loss at each cumulative level.
      5. Return mean over group losses (equal weights = 1/n_groups each).

    This encourages importance ordering: group 0 features must reconstruct
    most of the input on their own; later groups add residual corrections —
    using the SAME latent codes at all levels.

    Parameters
    ----------
    ae : MatryoshkaBatchTopKSAE
    x  : [B, D] input activations (on the correct device)

    Returns
    -------
    Scalar tensor: mean cumulative reconstruction loss
    """
    n_groups = ae.active_groups
    group_sizes = ae.group_sizes.tolist()[:n_groups]

    # Single full-dict encode
    f = ae.encode(x, use_threshold=False)          # [B, dict_size]

    # Split f and W_dec into per-group chunks
    W_dec_chunks = torch.split(ae.W_dec, group_sizes, dim=0)  # list of [gs, D]
    f_chunks = torch.split(f, group_sizes, dim=1)             # list of [B, gs]

    x_reconstruct = torch.zeros_like(x) + ae.b_dec            # start from bias
    group_losses = []

    for i in range(n_groups):
        x_reconstruct = x_reconstruct + f_chunks[i] @ W_dec_chunks[i]
        # temporal-saes uses sum-over-dims then batch-mean (not F.mse_loss)
        l2_loss = (x - x_reconstruct).pow(2).sum(dim=-1).mean()
        group_losses.append(l2_loss)

    return torch.stack(group_losses).mean()


# ---------------------------------------------------------------------------
# SpatialPatchTopKTrainer  (from spatial_patch_top_k.py)
# ---------------------------------------------------------------------------

class SpatialPatchTopKTrainer:
    """
    Trains MatryoshkaBatchTopKSAE with cumulative Matryoshka reconstruction loss
    + spatial InfoNCE contrastive loss on adjacent patch pairs.

    Follows the temporal-saes design:
      - Reconstruction: cumulative MSE at each Matryoshka group boundary
        (earlier groups receive more gradient signal — the nesting property)
      - Contrastive: HL-only (first dict_size//2 features), matching
        spatial_patch_top_k.py in T-SAE-Follow-Up
      - Encoder uses batch top-k (use_threshold=False) during training
    """

    def __init__(self, ae: MatryoshkaBatchTopKSAE, lr: float = 3e-4,
                 recon_alpha: float = 1.0, contrastive_alpha: float = 5.0,
                 temperature: float = 0.1, device: str = "cuda"):
        self.ae = ae.to(device)
        self.device = device
        self.recon_alpha = recon_alpha
        self.contrastive_alpha = contrastive_alpha
        self.temperature = temperature
        self.hl_split = ae.dict_size // 2  # always HL-only, matching T-SAE design
        self.opt = torch.optim.Adam(ae.parameters(), lr=lr)

    def step(self, x: torch.Tensor) -> dict:
        """x: [B, 2, D]  —  x[:,0]=anchor, x[:,1]=spatial neighbour"""
        x = x.to(self.device)
        x0, x1 = x[:, 0], x[:, 1]

        # Cumulative Matryoshka reconstruction loss (temporal-saes design)
        recon_loss = _matryoshka_recon_loss(self.ae, x0) + \
                     _matryoshka_recon_loss(self.ae, x1)

        # Encode for contrastive loss (batch top-k, no threshold)
        f0 = self.ae.encode(x0, use_threshold=False)
        f1 = self.ae.encode(x1, use_threshold=False)

        # Contrastive loss on HL features only (first dict_size//2)
        z0 = F.normalize(f0[:, :self.hl_split], dim=-1)
        z1 = F.normalize(f1[:, :self.hl_split], dim=-1)
        logits = (z0 @ z1.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)
        )

        loss = self.recon_alpha * recon_loss + self.contrastive_alpha * contrastive_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # keep decoder unit-norm (from temporal-saes)
        self.ae.W_dec.grad = _remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, self.ae.W_dec.grad.T,
            self.ae.activation_dim, self.ae.dict_size
        ).T
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.opt.step()

        self.ae.W_dec.data = _set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }


class MultiScaleSpatialTrainer:
    """
    Trains MatryoshkaBatchTopKSAE with cumulative Matryoshka reconstruction loss
    + multi-scale spatial InfoNCE on HL features.

    Follows the temporal-saes + T-SAE-Follow-Up design:
      - Reconstruction: cumulative MSE at each Matryoshka group boundary
      - Contrastive: HL-only (first dict_size//2), applied at each scale
        with individual temperatures and weights
      - Encoder uses batch top-k (use_threshold=False) during training
    """

    def __init__(
        self,
        ae: MatryoshkaBatchTopKSAE,
        scales: list[int] = [1, 2, 4],
        scale_weights: Optional[list[float]] = None,
        scale_temperatures: Optional[list[float]] = None,
        lr: float = 3e-4,
        recon_alpha: float = 1.0,
        contrastive_alpha: float = 5.0,
        device: str = "cuda",
    ):
        self.ae = ae.to(device)
        self.device = device
        self.scales = scales
        self.recon_alpha = recon_alpha
        self.contrastive_alpha = contrastive_alpha

        S = len(scales)
        raw_weights = scale_weights if scale_weights is not None else [1.0] * S
        total = sum(raw_weights)
        self.scale_weights = [w / total for w in raw_weights]
        self.scale_temperatures = (
            scale_temperatures if scale_temperatures is not None else [0.1] * S
        )
        assert len(self.scale_weights) == S
        assert len(self.scale_temperatures) == S

        # Contrastive loss on HL features only (first dict_size//2)
        self.hl_split = ae.dict_size // 2

        self.opt = torch.optim.Adam(ae.parameters(), lr=lr)

    def step(self, x: torch.Tensor) -> dict:
        """x: [B, S+1, D]  —  x[:,0]=anchor, x[:,1..]=scale neighbors"""
        x = x.to(self.device)
        B, S_plus_1, D = x.shape

        # Cumulative Matryoshka reconstruction loss on all tokens
        # Process each token position separately through the cumulative loss
        x_flat = x.view(B * S_plus_1, D)
        recon_loss = _matryoshka_recon_loss(self.ae, x_flat)

        # Encode all tokens with batch top-k for contrastive loss
        f_flat = self.ae.encode(x_flat, use_threshold=False)  # [B*(S+1), dict_size]
        f_all = f_flat.view(B, S_plus_1, -1)                  # [B, S+1, dict_size]

        # Multi-scale contrastive losses on HL features only
        f_anchor_hl = f_all[:, 0, :self.hl_split]             # [B, hl_split]
        z_anchor = F.normalize(f_anchor_hl, dim=-1)

        contrastive_loss = torch.tensor(0.0, device=self.device)
        S = len(self.scales)
        for s in range(S):
            if s + 1 >= S_plus_1:
                break
            f_nbr_hl = f_all[:, s + 1, :self.hl_split]
            z_nbr = F.normalize(f_nbr_hl, dim=-1)

            temp = self.scale_temperatures[s]
            logits = (z_anchor @ z_nbr.T) / temp
            labels = torch.arange(logits.size(0), device=self.device)
            loss_s = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )
            contrastive_loss = contrastive_loss + self.scale_weights[s] * loss_s

        loss = self.recon_alpha * recon_loss + self.contrastive_alpha * contrastive_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Decoder norm + gradient projection (from temporal-saes)
        self.ae.W_dec.grad = _remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, self.ae.W_dec.grad.T,
            self.ae.activation_dim, self.ae.dict_size
        ).T
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.opt.step()

        self.ae.W_dec.data = _set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }


# ---------------------------------------------------------------------------
# Spatial patch pair generation from pre-extracted tokens
# ---------------------------------------------------------------------------

def _grid_neighbors(side: int, r: int, c: int, mode: str = "8") -> list[tuple[int, int]]:
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == "8":
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [(r + dr, c + dc)
            for dr, dc in deltas
            if 0 <= r + dr < side and 0 <= c + dc < side]


def _patches_at_chebyshev_distance(
    h: int, w: int, r: int, c: int, d: int
) -> list[tuple[int, int]]:
    """Return all patches at exactly Chebyshev distance d from (r, c)."""
    neighbors = []
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if max(abs(dr), abs(dc)) != d:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                neighbors.append((rr, cc))
    return neighbors


def _patches_within_chebyshev_distance(
    h: int, w: int, r: int, c: int, d: int
) -> list[tuple[int, int]]:
    """Fallback: return all patches within Chebyshev distance d (excluding self)."""
    neighbors = []
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                neighbors.append((rr, cc))
    return neighbors


def _sample_patch_pairs(patch_tokens: torch.Tensor, pairs_per_image: int = 32,
                        neighbor_mode: str = "8") -> torch.Tensor:
    """
    patch_tokens: (B, N, D) pre-extracted tokens for a batch of images.
    Returns (B * pairs_per_image, 2, D) pair tensor.
    """
    bsz, n_patches, D = patch_tokens.shape
    side = int(math.sqrt(n_patches))
    assert side * side == n_patches, f"Expected square grid, got {n_patches} patches"

    grid = patch_tokens.view(bsz, side, side, D)
    pairs = []
    for b in range(bsz):
        for _ in range(pairs_per_image):
            r = random.randrange(side)
            c = random.randrange(side)
            nbrs = _grid_neighbors(side, r, c, mode=neighbor_mode)
            if not nbrs:
                continue
            rr, cc = random.choice(nbrs)
            pairs.append(torch.stack([grid[b, r, c], grid[b, rr, cc]], dim=0))  # (2, D)
    return torch.stack(pairs, dim=0) if pairs else torch.empty(0, 2, D)


def _sample_multiscale_patch_pairs(
    patch_tokens: torch.Tensor,
    scales: list[int],
    pairs_per_image: int = 32,
) -> torch.Tensor:
    """
    patch_tokens: (B, N, D)
    Returns (B * pairs_per_image, S+1, D) where dim1 = [anchor, scale1_nbr, scale2_nbr, ...]
    """
    bsz, n_patches, D = patch_tokens.shape
    side = int(math.sqrt(n_patches))
    assert side * side == n_patches, f"Expected square grid, got {n_patches} patches"

    S = len(scales)
    grid = patch_tokens.view(bsz, side, side, D)
    samples = []
    for b in range(bsz):
        for _ in range(pairs_per_image):
            r = random.randrange(side)
            c = random.randrange(side)
            anchor = grid[b, r, c]

            neighbors = []
            valid = True
            for d in scales:
                nbrs = _patches_at_chebyshev_distance(side, side, r, c, d)
                if not nbrs:
                    nbrs = _patches_within_chebyshev_distance(side, side, r, c, d)
                if not nbrs:
                    valid = False
                    break
                rr, cc = random.choice(nbrs)
                neighbors.append(grid[b, rr, cc])

            if not valid:
                continue
            samples.append(torch.stack([anchor] + neighbors, dim=0))  # (S+1, D)

    return torch.stack(samples, dim=0) if samples else torch.empty(0, S + 1, D)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_spatial_sae(
    patch_tokens: np.ndarray,
    n_steps: int = 500,
    dict_size: int = 1024,
    k: int = 64,
    group_fractions: Optional[list[float]] = None,
    lr: float = 3e-4,
    recon_alpha: float = 1.0,
    contrastive_alpha: float = 5.0,
    temperature: float = 0.1,
    batch_images: int = 32,
    pairs_per_image: int = 16,
    neighbor_mode: str = "8",
    scales: Optional[list[int]] = None,
    scale_weights: Optional[list[float]] = None,
    scale_temperatures: Optional[list[float]] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> MatryoshkaBatchTopKSAE:
    """
    Train a MatryoshkaBatchTopKSAE on spatial patch pairs drawn from
    pre-extracted DINOv2 patch tokens.

    Parameters
    ----------
    patch_tokens    : (N_images, N_patches, D_feat)  float32 array
    n_steps         : number of gradient steps
    dict_size       : SAE dictionary size
    k               : batch top-k sparsity
    group_fractions : Matryoshka group fractions (default [0.25]*4)
    scales          : Chebyshev distances for multi-scale loss (default [1,2,4]).
                      Pass [1] to use single-scale.
    scale_weights   : per-scale loss weights (default [1.0, 0.5, 0.2])
    scale_temperatures : per-scale InfoNCE temperature (default [0.1, 0.15, 0.2])

    Both trainers use cumulative Matryoshka reconstruction losses (temporal-saes
    design) and apply contrastive loss on HL features only (first dict_size//2).

    Returns
    -------
    Trained MatryoshkaBatchTopKSAE (on device, eval mode)
    """
    if group_fractions is None:
        group_fractions = [0.25, 0.25, 0.25, 0.25]
    assert abs(sum(group_fractions) - 1.0) < 1e-5

    # Multi-scale defaults (Wendy's settings)
    if scales is None:
        scales = [1, 2, 4]
    if scale_weights is None:
        scale_weights = [1.0, 0.5, 0.2] if len(scales) == 3 else [1.0] * len(scales)
    if scale_temperatures is None:
        scale_temperatures = [0.1, 0.15, 0.2] if len(scales) == 3 else [temperature] * len(scales)

    group_sizes = [int(f * dict_size) for f in group_fractions[:-1]]
    group_sizes.append(dict_size - sum(group_sizes))

    N, n_patches, D = patch_tokens.shape
    tokens_t = torch.from_numpy(patch_tokens).float()  # keep on CPU, move per-batch

    ae = MatryoshkaBatchTopKSAE(
        activation_dim=D, dict_size=dict_size, k=k, group_sizes=group_sizes
    ).to(device)

    if len(scales) > 1 or scales[0] != 1:
        trainer = MultiScaleSpatialTrainer(
            ae=ae, scales=scales, scale_weights=scale_weights,
            scale_temperatures=scale_temperatures,
            lr=lr, recon_alpha=recon_alpha,
            contrastive_alpha=contrastive_alpha, device=device,
        )
    else:
        trainer = SpatialPatchTopKTrainer(
            ae=ae, lr=lr, recon_alpha=recon_alpha,
            contrastive_alpha=contrastive_alpha,
            temperature=scale_temperatures[0], device=device,
        )

    indices = list(range(N))
    step = 0
    log_interval = max(1, n_steps // 20)

    while step < n_steps:
        random.shuffle(indices)
        for start in range(0, N, batch_images):
            if step >= n_steps:
                break
            batch_idx = indices[start:start + batch_images]
            if not batch_idx:
                continue
            batch = tokens_t[batch_idx]  # (b, N_patches, D)

            if isinstance(trainer, MultiScaleSpatialTrainer):
                pairs = _sample_multiscale_patch_pairs(
                    batch, scales=scales, pairs_per_image=pairs_per_image
                )
            else:
                pairs = _sample_patch_pairs(batch, pairs_per_image=pairs_per_image,
                                             neighbor_mode=neighbor_mode)

            if pairs.shape[0] == 0:
                continue
            losses = trainer.step(pairs)
            if verbose and step % log_interval == 0:
                print(f"  [spatial_sae step {step}/{n_steps}] "
                      f"loss={losses['loss']:.4f}  "
                      f"recon={losses['recon_loss']:.4f}  "
                      f"contrastive={losses['contrastive_loss']:.4f}")
            step += 1

    ae.eval()
    return ae


@torch.no_grad()
def encode_patches(
    ae: MatryoshkaBatchTopKSAE,
    patch_tokens: np.ndarray,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode pre-extracted patch tokens patch-by-patch.

    Parameters
    ----------
    patch_tokens : (N_images, N_patches, D_feat)
    Returns
    -------
    codes : (N_images, N_patches, dict_size)  float32 numpy array
    """
    ae = ae.to(device).eval()
    N, P, D = patch_tokens.shape
    codes = np.zeros((N, P, ae.dict_size), dtype=np.float32)
    tokens_t = torch.from_numpy(patch_tokens).float()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = tokens_t[start:end].to(device)              # (b, P, D)
        flat = batch.reshape(-1, D)                          # (b*P, D)
        f = ae.encode(flat, use_threshold=False)             # (b*P, dict_size)
        codes[start:end] = f.reshape(end - start, P, ae.dict_size).cpu().numpy()

    return codes


@torch.no_grad()
def encode_patches_per_group_topk(
    ae: MatryoshkaBatchTopKSAE,
    patch_tokens: np.ndarray,
    device: str = "cuda",
    batch_size: int = 64,
    k_per_group: int | None = None,
) -> np.ndarray:
    """
    Encode patch tokens using per-group Top-K instead of global threshold.

    Each Matryoshka group contributes exactly k_per_group active features
    (default: k // n_groups), enforcing equal representation from every group.
    This makes the Matryoshka hierarchy meaningful at inference time.

    Parameters
    ----------
    patch_tokens : (N_images, N_patches, D_feat)
    k_per_group  : top-k per group; defaults to ae.k // n_groups
    Returns
    -------
    codes : (N_images, N_patches, dict_size)  float32 numpy array
    """
    ae = ae.to(device).eval()
    N, P, D = patch_tokens.shape
    group_indices = ae.group_indices          # list: [0, g1, g2, …, dict_size]
    n_groups = ae.active_groups
    k_global = int(ae.k.item())
    if k_per_group is None:
        k_per_group = max(1, k_global // n_groups)

    codes = np.zeros((N, P, ae.dict_size), dtype=np.float32)
    tokens_t = torch.from_numpy(patch_tokens).float()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = tokens_t[start:end].to(device)       # (b, P, D)
        flat = batch.reshape(-1, D)                   # (b*P, D)

        post_relu = torch.relu((flat - ae.b_dec) @ ae.W_enc + ae.b_enc)
        f = torch.zeros_like(post_relu)

        for gi in range(n_groups):
            s = group_indices[gi]
            e = group_indices[gi + 1]
            grp = post_relu[:, s:e]                  # (b*P, group_size)
            kk = min(k_per_group, grp.shape[1])
            topk = grp.topk(kk, dim=1)
            f[:, s:e].scatter_(1, topk.indices, topk.values)

        codes[start:end] = f.reshape(end - start, P, ae.dict_size).cpu().numpy()

    return codes


def mean_pool_codes(codes: np.ndarray) -> np.ndarray:
    """
    Mean-pool spatial dimension: (N, N_patches, dict_size) -> (N, dict_size)
    Compatible with step3_run_analysis.py Ridge regression pipeline.
    """
    return codes.mean(axis=1)
