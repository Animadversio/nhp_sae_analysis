"""
Sparse Autoencoder (SAE) and Temporal SAE (T-SAE) Feature Extraction
=====================================================================
T-SAE follows: "Temporal Sparse Autoencoders: Leveraging the Sequential
Nature of Language for Interpretability"

Key design of T-SAE
--------------------
- Matryoshka dictionary split into G groups
- Group 0 (high-level): reconstruction loss + temporal consistency loss
- Group 1+ (low-level): cumulative reconstruction loss ONLY
- Temporal loss = |f0(x_t) - f0(x_{t-1})| * cos_sim(x_t, x_{t-1})
  (or CLIP-style contrastive on Group 0)

Faithfully implemented details from the paper codebase:
  1. Batch TopK sparsity (flatten across batch, keep k*B activations)
  2. remove_gradient_parallel_to_decoder_directions before each update
  3. Auxiliary loss on dead neurons (neurons not fired in >threshold steps)
  4. LR warmup schedule
  5. Geometric median initialisation of b_dec
  6. Decoder unit-norm constraint (enforced after every step)
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


# ---------------------------------------------------------------------------
# Decoder norm utilities (mirrors trainer.py from the paper)
# ---------------------------------------------------------------------------

def _set_decoder_unit_norm(W_dec: torch.Tensor) -> torch.Tensor:
    """
    W_dec: (d_dict, d_input)  — normalise each row to unit norm.
    Equivalent to the paper's set_decoder_norm_to_unit_norm(W_dec.T, ...).T
    """
    norms = W_dec.norm(dim=1, keepdim=True).clamp(min=torch.finfo(W_dec.dtype).eps)
    return W_dec / norms


@torch.no_grad()
def _remove_gradient_parallel_to_decoder(
    W_dec: torch.Tensor,      # (d_dict, d_input)  current weights
    W_dec_grad: torch.Tensor, # (d_dict, d_input)  gradient
) -> torch.Tensor:
    """
    Remove the component of the gradient parallel to each decoder direction.
    This prevents the gradient step from changing the norm of decoder columns,
    keeping the unit-norm constraint stable during training.

    Mirrors remove_gradient_parallel_to_decoder_directions from the paper.
    """
    normed = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-6)  # (d_dict, d_input)
    # Dot product of grad with normed direction, per dictionary atom
    parallel = (W_dec_grad * normed).sum(dim=1, keepdim=True)   # (d_dict, 1)
    W_dec_grad = W_dec_grad - parallel * normed
    return W_dec_grad


def _geometric_median(points: torch.Tensor,
                       max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """Weiszfeld algorithm for geometric median. Used to initialise b_dec."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        dists = torch.norm(points - guess, dim=1).clamp(min=1e-6)
        weights = 1.0 / dists
        weights /= weights.sum()
        new_guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(new_guess - guess) < tol:
            break
        guess = new_guess
    return guess


# ---------------------------------------------------------------------------
# Standard SAE  (Batch TopK)
# ---------------------------------------------------------------------------

class SAE(nn.Module):
    """
    Batch TopK Sparse Autoencoder.
    Batch TopK: flatten activations across the batch, keep k*B total,
    giving a stable average sparsity of k per sample.
    """

    def __init__(self, d_input: int = 768, d_dict: int = 4096, k: int = 32):
        super().__init__()
        self.d_input = d_input
        self.d_dict  = d_dict
        self.register_buffer('k', torch.tensor(k, dtype=torch.int))
        self.register_buffer('threshold', torch.tensor(-1.0))
        # Dead-neuron tracking
        self.register_buffer('num_tokens_since_fired',
                              torch.zeros(d_dict, dtype=torch.long))
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = d_input // 2   # heuristic from paper appendix B.1

        self.W_enc = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_dict, d_input))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        nn.init.kaiming_uniform_(self.W_dec)
        self.W_dec.data = _set_decoder_unit_norm(self.W_dec)
        self.W_enc.data = self.W_dec.data.clone()  # W_enc = W_dec (tied init)

    def encode(self, x: torch.Tensor,
               return_pre: bool = False,
               use_threshold: bool = False):
        pre = F.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        if use_threshold and self.threshold >= 0:
            z = pre * (pre > self.threshold)
        else:
            flat = pre.flatten()
            topk = flat.topk(self.k.item() * x.shape[0], sorted=False)
            z = torch.zeros_like(flat).scatter_(-1, topk.indices, topk.values)
            z = z.reshape(pre.shape)
        return (z, pre) if return_pre else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def get_auxiliary_loss(self, residual: torch.Tensor,
                            pre: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary reconstruction loss on dead neurons (paper appendix B.1).
        Prevents dictionary collapse by keeping dead atoms alive.
        """
        dead = self.num_tokens_since_fired >= self.dead_feature_threshold
        if not dead.any():
            return torch.tensor(0.0, device=residual.device, dtype=residual.dtype)

        k_aux = min(self.top_k_aux, int(dead.sum()))
        auxk_acts = torch.where(dead[None], pre, torch.tensor(-torch.inf, device=pre.device))
        vals, idxs = auxk_acts.topk(k_aux, sorted=False)

        aux_z = torch.zeros_like(pre).scatter_(-1, idxs, vals)
        x_aux = aux_z @ self.W_dec          # no b_dec (matches paper)

        l2_aux = (residual.float() - x_aux.float()).pow(2).sum(-1).mean()
        # Normalise by variance of residual (OpenAI normalisation)
        mu = residual.mean(0, keepdim=True)
        denom = (residual.float() - mu.float()).pow(2).sum(-1).mean()
        return (l2_aux / denom.clamp(min=1e-6)).nan_to_num(0.0)

    @torch.no_grad()
    def update_firing_stats(self, z: torch.Tensor, batch_size: int):
        fired = (z.sum(0) > 0)
        self.num_tokens_since_fired += batch_size
        self.num_tokens_since_fired[fired] = 0

    @torch.no_grad()
    def normalise_decoder(self):
        self.W_dec.data = _set_decoder_unit_norm(self.W_dec)

    @torch.no_grad()
    def update_threshold(self, z: torch.Tensor, beta: float = 0.999):
        active = z[z > 0]
        v = active.min().item() if active.numel() > 0 else 0.0
        if self.threshold < 0:
            self.threshold.fill_(v)
        else:
            self.threshold.mul_(beta).add_((1 - beta) * v)


# ---------------------------------------------------------------------------
# Temporal SAE  (Matryoshka + temporal loss only on Group 0)
# ---------------------------------------------------------------------------

class TSAE(nn.Module):
    """
    Temporal Sparse Autoencoder with Matryoshka structure.

    Key design (faithful to TemporalMatryoshkaBatchTopKSAE):
      - Group 0: reconstruction(x_t using group 0 alone) + temporal loss
      - Group i≥1: CUMULATIVE reconstruction(x_t using groups 0..i), NO temporal loss
      - Temporal loss only on Group 0 features (f_chunks[0])
      - W_enc initialised as W_dec (tied), decoder cols unit-norm
      - Auxiliary loss on dead neurons
      - Gradient parallel to decoder directions removed each step
    """

    def __init__(
        self,
        d_input:         int         = 768,
        d_dict:          int         = 4096,
        k:               int         = 32,
        group_fractions: list[float] = None,
        group_weights:   list[float] = None,
        temp_alpha:      float       = 0.1,
        contrastive:     bool        = False,
    ):
        super().__init__()
        self.d_input     = d_input
        self.d_dict      = d_dict
        self.temp_alpha  = temp_alpha
        self.contrastive = contrastive

        if group_fractions is None:
            group_fractions = [0.5, 0.5]
        assert abs(sum(group_fractions) - 1.0) < 1e-5

        sizes = [int(f * d_dict) for f in group_fractions[:-1]]
        sizes.append(d_dict - sum(sizes))
        self.group_sizes = sizes
        self.n_groups    = len(sizes)

        if group_weights is None:
            group_weights = [1.0 / self.n_groups] * self.n_groups
        self.group_weights = group_weights

        ends = []
        s = 0
        for gs in sizes:
            s += gs
            ends.append(s)
        self.group_ends = ends

        self.register_buffer('k', torch.tensor(k, dtype=torch.int))
        self.register_buffer('threshold', torch.tensor(-1.0))
        self.register_buffer('num_tokens_since_fired',
                              torch.zeros(d_dict, dtype=torch.long))
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = d_input // 2

        self.W_enc = nn.Parameter(torch.empty(d_input, d_dict))
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.empty(d_dict, d_input))
        self.b_dec = nn.Parameter(torch.zeros(d_input))

        nn.init.kaiming_uniform_(self.W_dec)
        self.W_dec.data = _set_decoder_unit_norm(self.W_dec)
        self.W_enc.data = self.W_dec.data.clone()  # tied init

    def encode(self, x: torch.Tensor,
               return_pre: bool = False,
               use_threshold: bool = False):
        """x: (B, d_input)"""
        pre = F.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        if use_threshold and self.threshold >= 0:
            z = pre * (pre > self.threshold)
        else:
            flat = pre.flatten()
            topk = flat.topk(self.k.item() * x.shape[0], sorted=False)
            z = torch.zeros_like(flat).scatter_(-1, topk.indices, topk.values)
            z = z.reshape(pre.shape)
        return (z, pre) if return_pre else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def compute_loss(self, x_pair: torch.Tensor,
                     auxk_alpha: float = 1/32) -> dict:
        """
        x_pair: (B, 2, d_input)
                [:, 0] = x_t  (current),  [:, 1] = x_{t-1} (previous)

        Loss:
          Group 0:  recon_0(x_t) + temp_alpha * temporal_loss(f0_t, f0_{t-1})
          Group i≥1: recon_0..i(x_t)   [cumulative, no temporal term]
          + auxk_alpha * auxiliary_dead_neuron_loss
        """
        x_t   = x_pair[:, 0]
        x_tm1 = x_pair[:, 1]

        z_t,   pre_t   = self.encode(x_t,   return_pre=True)
        z_tm1          = self.encode(x_tm1,  return_pre=False)

        f_chunks      = torch.split(z_t,   self.group_sizes, dim=1)
        f_prev_chunks = torch.split(z_tm1, self.group_sizes, dim=1)
        W_chunks      = torch.split(self.W_dec, self.group_sizes, dim=0)

        # ── Group 0: independent recon + temporal loss ─────────────────
        x_reconstruct = self.b_dec + f_chunks[0] @ W_chunks[0]
        recon_0 = ((x_t - x_reconstruct).pow(2).sum(-1).mean()
                   * self.group_weights[0])
        total_recon = recon_0

        cos_sim = F.cosine_similarity(x_t, x_tm1, dim=-1).clamp(min=0)

        if self.contrastive:
            logits = f_chunks[0] @ f_prev_chunks[0].T
            labels = torch.arange(x_t.shape[0], device=x_t.device)
            temp_loss = (F.cross_entropy(logits, labels) +
                         F.cross_entropy(logits.T, labels)) / 2
        else:
            # Paper eq: |f0(x_t) - f0(x_{t-1})| * cos_sim * group_weight
            temp_loss = (torch.abs(f_chunks[0] - f_prev_chunks[0]).sum(-1)
                         * cos_sim * self.group_weights[0]).mean()

        # ── Groups 1+: cumulative recon only ───────────────────────────
        for i in range(1, self.n_groups):
            x_reconstruct = x_reconstruct + f_chunks[i] @ W_chunks[i]
            recon_i = ((x_t - x_reconstruct).pow(2).sum(-1).mean()
                       * self.group_weights[i])
            total_recon = total_recon + recon_i

        # ── Auxiliary loss on dead neurons ─────────────────────────────
        residual = (x_t - x_reconstruct).detach()
        auxk_loss = self._auxiliary_loss(residual, pre_t)

        loss = total_recon + self.temp_alpha * temp_loss + auxk_alpha * auxk_loss

        return {
            'loss':       loss,
            'recon_loss': total_recon.detach(),
            'temp_loss':  temp_loss.detach(),
            'auxk_loss':  auxk_loss.detach(),
            'z_t':        z_t,
            'pre_t':      pre_t,
        }

    def _auxiliary_loss(self, residual: torch.Tensor,
                         pre: torch.Tensor) -> torch.Tensor:
        dead = self.num_tokens_since_fired >= self.dead_feature_threshold
        if not dead.any():
            return torch.tensor(0.0, device=residual.device, dtype=residual.dtype)
        k_aux = min(self.top_k_aux, int(dead.sum()))
        auxk_acts = torch.where(dead[None], pre,
                                torch.tensor(-torch.inf, device=pre.device))
        vals, idxs = auxk_acts.topk(k_aux, sorted=False)
        aux_z = torch.zeros_like(pre).scatter_(-1, idxs, vals)
        x_aux = aux_z @ self.W_dec
        l2_aux = (residual.float() - x_aux.float()).pow(2).sum(-1).mean()
        mu = residual.mean(0, keepdim=True)
        denom = (residual.float() - mu.float()).pow(2).sum(-1).mean()
        return (l2_aux / denom.clamp(min=1e-6)).nan_to_num(0.0)

    def get_group_codes(self, z: torch.Tensor) -> list[torch.Tensor]:
        """Split z into per-group codes: group 0 = high-level, 1+ = low-level."""
        return list(torch.split(z, self.group_sizes, dim=-1))

    @torch.no_grad()
    def update_firing_stats(self, z: torch.Tensor, batch_size: int):
        fired = (z.sum(0) > 0)
        self.num_tokens_since_fired += batch_size
        self.num_tokens_since_fired[fired] = 0

    @torch.no_grad()
    def normalise_decoder(self):
        self.W_dec.data = _set_decoder_unit_norm(self.W_dec)

    @torch.no_grad()
    def update_threshold(self, z: torch.Tensor, beta: float = 0.999):
        active = z[z > 0]
        v = active.min().item() if active.numel() > 0 else 0.0
        if self.threshold < 0:
            self.threshold.fill_(v)
        else:
            self.threshold.mul_(beta).add_((1 - beta) * v)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _get_lr_fn(total_steps: int, warmup_steps: int):
    """Linear warmup then constant (simplified from paper's get_lr_schedule)."""
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    return lr_fn


def _train_sae(model, data, n_epochs, lr, batch_size, warmup_steps,
               auxk_alpha, threshold_start_step, device, verbose):
    model = model.to(device)
    data  = data.to(device)

    # Initialise b_dec with geometric median of data
    with torch.no_grad():
        sample = data[torch.randperm(len(data))[:min(1000, len(data))]]
        model.b_dec.data = _geometric_median(sample).to(model.b_dec.dtype)

    total_steps = n_epochs * (len(data) // batch_size + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, _get_lr_fn(total_steps, warmup_steps))

    losses = []
    step = 0
    for epoch in range(n_epochs):
        perm = torch.randperm(len(data), device=device)
        ep_recon = ep_auxk = 0.0; nb = 0
        for s in range(0, len(data), batch_size):
            xb = data[perm[s:s+batch_size]]
            opt.zero_grad()

            z, pre = model.encode(xb, return_pre=True)
            xhat   = model.decode(z)
            recon  = (xb - xhat).pow(2).sum(-1).mean()
            auxk   = model.get_auxiliary_loss((xb - xhat).detach(), pre)
            loss   = recon + auxk_alpha * auxk
            loss.backward()

            # Remove gradient component parallel to decoder directions
            if model.W_dec.grad is not None:
                model.W_dec.grad.data = _remove_gradient_parallel_to_decoder(
                    model.W_dec.data, model.W_dec.grad.data
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            model.normalise_decoder()

            model.update_firing_stats(z.detach(), xb.shape[0])
            if step > threshold_start_step:
                model.update_threshold(z.detach())

            ep_recon += recon.item(); ep_auxk += auxk.item(); nb += 1; step += 1

        avg = ep_recon / max(nb, 1); losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  SAE epoch {epoch+1:3d}/{n_epochs}  "
                  f"recon={avg:.4f}  auxk={ep_auxk/max(nb,1):.4f}")
    return losses


def _make_pairs(data_flat: torch.Tensor) -> torch.Tensor:
    """
    (N_tokens, D) → (N_tokens-1, 2, D)
    [:, 0] = x_t (current), [:, 1] = x_{t-1} (previous)
    Matches the paper's buffer: stack([curr, prev], dim=1).
    """
    return torch.stack([data_flat[1:], data_flat[:-1]], dim=1)


def _train_tsae(model, data_flat, n_epochs, lr, batch_size, warmup_steps,
                auxk_alpha, threshold_start_step, device, verbose):
    model = model.to(device)
    pairs = _make_pairs(data_flat).to(device)   # (N-1, 2, D)

    # Initialise b_dec with geometric median of x_t tokens
    with torch.no_grad():
        sample_idx = torch.randperm(len(pairs))[:min(1000, len(pairs))]
        sample = pairs[sample_idx, 0]
        model.b_dec.data = _geometric_median(sample).to(model.b_dec.dtype)

    total_steps = n_epochs * (len(pairs) // batch_size + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, _get_lr_fn(total_steps, warmup_steps))

    losses = []
    step = 0
    for epoch in range(n_epochs):
        perm = torch.randperm(len(pairs), device=device)
        ep_tot = ep_rec = ep_tmp = ep_aux = 0.0; nb = 0
        for s in range(0, len(pairs), batch_size):
            xb = pairs[perm[s:s+batch_size]]
            opt.zero_grad()
            out = model.compute_loss(xb, auxk_alpha=auxk_alpha)
            out['loss'].backward()

            # Remove gradient component parallel to decoder directions
            if model.W_dec.grad is not None:
                model.W_dec.grad.data = _remove_gradient_parallel_to_decoder(
                    model.W_dec.data, model.W_dec.grad.data
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            model.normalise_decoder()

            model.update_firing_stats(out['z_t'].detach(), xb.shape[0])
            if step > threshold_start_step:
                model.update_threshold(out['z_t'].detach())

            ep_tot += out['loss'].item()
            ep_rec += out['recon_loss'].item()
            ep_tmp += out['temp_loss'].item()
            ep_aux += out['auxk_loss'].item()
            nb += 1; step += 1

        avg = ep_tot / max(nb, 1); losses.append(avg)
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  T-SAE epoch {epoch+1:3d}/{n_epochs}  "
                  f"total={avg:.4f}  recon={ep_rec/max(nb,1):.4f}  "
                  f"temp={ep_tmp/max(nb,1):.4f}  auxk={ep_aux/max(nb,1):.4f}")
    return losses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_and_encode(
    model_type:           Literal['SAE', 'TSAE'] = 'SAE',
    features:             np.ndarray = None,     # (N_images, N_patches, D)
    d_dict:               int   = 4096,
    k:                    int   = 32,
    n_epochs:             int   = 50,
    lr:                   float = 2e-4,
    batch_size:           int   = 256,
    warmup_steps:         int   = 200,
    auxk_alpha:           float = 1/32,
    threshold_start_step: int   = 200,
    temp_alpha:           float = 0.1,
    contrastive:          bool  = False,
    group_fractions:      list  = None,           # default [0.5, 0.5]
    group_weights:        list  = None,
    device:               str   = 'cpu',
    verbose:              bool  = True,
) -> tuple[np.ndarray, SAE | TSAE, list]:
    """
    Train SAE or T-SAE on DINOv2 patch tokens, return sparse codes.

    For TSAE, group 0 codes are high-level (temporally consistent),
    group 1+ are low-level (reconstruction detail).
    Access via: model.get_group_codes(z_tensor)

    Returns
    -------
    codes  : (N_images, N_patches, d_dict)
    model  : trained model
    losses : per-epoch total losses
    """
    N, P, D = features.shape
    flat    = torch.tensor(features.reshape(N * P, D), dtype=torch.float32)

    if model_type == 'SAE':
        model  = SAE(d_input=D, d_dict=d_dict, k=k)
        losses = _train_sae(model, flat, n_epochs, lr, batch_size,
                             warmup_steps, auxk_alpha, threshold_start_step,
                             device, verbose)

    elif model_type == 'TSAE':
        model  = TSAE(d_input=D, d_dict=d_dict, k=k,
                      group_fractions=group_fractions,
                      group_weights=group_weights,
                      temp_alpha=temp_alpha,
                      contrastive=contrastive)
        losses = _train_tsae(model, flat, n_epochs, lr, batch_size,
                              warmup_steps, auxk_alpha, threshold_start_step,
                              device, verbose)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model.eval().to(device)
    with torch.no_grad():
        codes_flat = model.encode(flat.to(device), use_threshold=True).cpu().numpy()

    return codes_flat.reshape(N, P, d_dict), model.cpu(), losses


def mean_pool_codes(codes: np.ndarray) -> np.ndarray:
    """(N, P, d_dict) → (N, d_dict)"""
    return codes.mean(axis=1)
