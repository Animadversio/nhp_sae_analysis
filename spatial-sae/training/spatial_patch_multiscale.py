"""
spatial_patch_multiscale.py

Spatial SAE trainer with multi-scale contrastive loss.

Instead of a single adjacency distance, applies contrastive loss at
multiple spatial scales simultaneously:
  - scale 1 (+-1): direct neighbors → captures fine-grained local coherence
  - scale 2 (+-2): patches 2 steps apart → captures mid-range coherence
  - scale 4 (+-4): patches 4 steps apart → captures coarse/global coherence

Following the T-SAE design:
  - Contrastive loss applied only to the HIGH-LEVEL group (first dict_size//2 features)
  - Each scale gets its own temperature and weight
  - Reconstruction loss applied to all features as usual

Input format:
  x: [B, S+1, D]
     x[:, 0]   = anchor patch
     x[:, 1]   = scale-1 neighbor  (+-1)
     x[:, 2]   = scale-2 neighbor  (+-2)
     x[:, 3]   = scale-4 neighbor  (+-4)
  where S = number of scales (default 3)

The data pipeline (vision_patch_pairs_multiscale.py) must produce
this format by sampling neighbors at each scale for every anchor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z0: torch.Tensor, z1: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) loss between two sets of L2-normalized vectors.
    z0, z1: [B, D], already L2-normalized
    Treats (z0[i], z1[i]) as positive pairs, all other combinations as negatives.
    """
    logits = (z0 @ z1.T) / temperature          # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits,   labels) +
        F.cross_entropy(logits.T, labels)
    )
    return loss


class MultiScaleSpatialTrainer:
    """
    Trains a Matryoshka SAE with:
      1. Reconstruction loss on all features
      2. Multi-scale spatial contrastive loss on high-level features only

    Args:
        ae:                  MatryoshkaBatchTopKSAE instance
        scales:              list of adjacency distances, e.g. [1, 2, 4]
        scale_weights:       weight for contrastive loss at each scale
                             (default: equal weights, sum to contrastive_alpha)
        scale_temperatures:  temperature for NT-Xent at each scale
                             (default: same temperature for all scales)
        recon_alpha:         weight for reconstruction loss
        contrastive_alpha:   total weight for contrastive losses (split across scales)
        lr:                  learning rate
        device:              "cuda" or "cpu"
    """

    def __init__(
        self,
        ae,
        scales: list[int] = [1, 2, 4],
        scale_weights: list[float] | None = None,
        scale_temperatures: list[float] | None = None,
        recon_alpha: float = 1.0,
        contrastive_alpha: float = 3.0,
        lr: float = 3e-4,
        device: str = "cuda",
    ):
        self.ae = ae.to(device)
        self.device = device
        self.scales = scales
        self.recon_alpha = recon_alpha
        self.contrastive_alpha = contrastive_alpha

        n_scales = len(scales)

        # Default: equal weight across scales, sum = contrastive_alpha
        if scale_weights is None:
            scale_weights = [1.0 / n_scales] * n_scales
        # Normalize so they sum to 1 (total weight = contrastive_alpha)
        total = sum(scale_weights)
        self.scale_weights = [w / total for w in scale_weights]

        # Default: same temperature for all scales
        if scale_temperatures is None:
            scale_temperatures = [0.1] * n_scales
        self.scale_temperatures = scale_temperatures

        # High-level split: contrastive loss only on first dict_size//2 features
        dict_size = ae.dict_size if hasattr(ae, "dict_size") else None
        if dict_size is None:
            # Try to infer from encoder weight shape
            for p in ae.parameters():
                if len(p.shape) == 2:
                    dict_size = max(p.shape)
                    break
        self.hl_split = dict_size // 2

        self.opt = torch.optim.Adam(ae.parameters(), lr=lr)

        print(f"[MultiScaleSpatialTrainer] scales={scales}")
        print(f"  scale_weights={[f'{w:.3f}' for w in self.scale_weights]}")
        print(f"  scale_temperatures={scale_temperatures}")
        print(f"  hl_split={self.hl_split} (contrastive loss on first {self.hl_split} features)")

    def step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        One training step.

        Args:
            x: [B, S+1, D]
               x[:, 0]     = anchor patch tokens
               x[:, 1]     = scale-1 neighbor tokens
               x[:, 2]     = scale-2 neighbor tokens  (if n_scales >= 2)
               x[:, 3]     = scale-4 neighbor tokens  (if n_scales >= 3)

        Returns:
            dict with loss values for logging
        """
        x = x.to(self.device)
        anchor = x[:, 0]   # [B, D]

        # ── 1. Reconstruction loss ───────────────────────────
        # Encode all tokens (anchor + all neighbors) and reconstruct
        # Flatten to [B*(S+1), D], encode, decode, reshape
        B, S_plus_1, D = x.shape
        x_flat = x.view(B * S_plus_1, D)
        f_flat  = self.ae.encode(x_flat)          # [B*(S+1), dict_size]
        x_hat   = self.ae.decode(f_flat)          # [B*(S+1), D]
        recon_loss = F.mse_loss(x_hat, x_flat)

        # Reshape features back
        f_all = f_flat.view(B, S_plus_1, -1)      # [B, S+1, dict_size]
        f_anchor = f_all[:, 0]                     # [B, dict_size]

        # High-level features only (for contrastive loss)
        f_anchor_hl = f_anchor[:, :self.hl_split]  # [B, hl_split]

        # ── 2. Multi-scale contrastive loss ─────────────────
        contrastive_losses = []

        for scale_idx, (scale, weight, temp) in enumerate(
            zip(self.scales, self.scale_weights, self.scale_temperatures)
        ):
            if scale_idx + 1 >= S_plus_1:
                break  # not enough scales in this batch

            f_neighbor = f_all[:, scale_idx + 1]           # [B, dict_size]
            f_neighbor_hl = f_neighbor[:, :self.hl_split]  # [B, hl_split]

            # L2 normalize
            z0 = F.normalize(f_anchor_hl,   dim=-1)
            z1 = F.normalize(f_neighbor_hl, dim=-1)

            loss_s = nt_xent_loss(z0, z1, temperature=temp)
            contrastive_losses.append(weight * loss_s)

        total_contrastive = sum(contrastive_losses) if contrastive_losses \
            else torch.tensor(0.0, device=self.device, requires_grad=True)

        # ── 3. Total loss ────────────────────────────────────
        loss = self.recon_alpha * recon_loss + self.contrastive_alpha * total_contrastive

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        log = {
            "loss":              loss.detach(),
            "recon_loss":        recon_loss.detach(),
            "contrastive_total": total_contrastive.detach(),
        }
        for i, (scale, c_loss) in enumerate(
            zip(self.scales, contrastive_losses)
        ):
            log[f"contrastive_scale{scale}"] = c_loss.detach()

        return log