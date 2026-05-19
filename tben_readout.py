"""
TBEn (Transformer Brain Encoder) Cross-Attention Readout
=========================================================
Implements the readout method from:
    Adeli et al. 2026 — "Transformer brain encoders explain human
    high-level visual responses"

Key idea
--------
Instead of flattening all patch tokens and doing Ridge regression,
each neuron (or neuron population) gets a *learnable query vector*
that attends over the image patch tokens via cross-attention.
The attended representation is then linearly mapped to firing rate.

This is more expressive than Ridge because:
  - The spatial weighting is *input-content-dependent* (not fixed)
  - Positional encoding lets queries route by location OR content
  - Each neuron learns which patches are relevant independently

Adaptation from the original
-----------------------------
Original TBEn targets fMRI vertices grouped by ROI.
Here we target individual spike-sorted units (single neurons / MUA).
Because unit count can be large (100–600 per session), we offer two
modes:
  - 'per_unit'  : one query per unit  (most expressive, slowest)
  - 'shared'    : units share queries via a final linear projection
                  (faster, similar to the ROI-query design of TBEn)

Usage
-----
    from tben_readout import TBEnReadout, train_tben

    # features : (N_images, N_patches, D_feat)  — patch tokens from DINOv2
    # responses: (N_images, N_units)             — mean firing rates

    model = TBEnReadout(d_feat=768, n_units=300, n_queries=32, mode='shared')
    results = train_tben(model, features, responses,
                         n_epochs=50, lr=1e-3, device='cpu')
    print(results['test_r2'].mean())   # mean R² across units
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from typing import Literal


# ---------------------------------------------------------------------------
# Positional Encoding (2-D sinusoidal for patch grid)
# ---------------------------------------------------------------------------

def make_2d_sincos_pos_embed(d_model: int, grid_size: int) -> torch.Tensor:
    """
    Build (grid_size**2, d_model) 2D sincos position embeddings.
    Compatible with ViT-B/14 patch grid (14×14 = 196 patches).
    """
    assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sincos"
    h = w = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid   = torch.stack(torch.meshgrid(grid_h, grid_w, indexing='ij'), dim=0)  # (2, H, W)
    grid   = grid.reshape(2, -1).T   # (H*W, 2)

    d_half = d_model // 4
    omega  = 1.0 / (10000 ** (torch.arange(d_half, dtype=torch.float32) / d_half))
    out_h  = torch.einsum('n,d->nd', grid[:, 0], omega)   # (N, d_half)
    out_w  = torch.einsum('n,d->nd', grid[:, 1], omega)
    emb    = torch.cat([out_h.sin(), out_h.cos(),
                        out_w.sin(), out_w.cos()], dim=-1)  # (N, d_model)
    return emb   # not a parameter — will be added as a buffer


# ---------------------------------------------------------------------------
# Core TBEn readout
# ---------------------------------------------------------------------------

class TBEnReadout(nn.Module):
    """
    Cross-attention readout from patch tokens to neuron responses.

    Parameters
    ----------
    d_feat     : feature dimension of each patch token (e.g. 768 for DINOv2 ViT-B)
    n_units    : number of output neurons
    n_queries  : number of learnable query vectors (only used when mode='shared')
    n_heads    : number of attention heads
    mode       : 'per_unit' — one query per neuron (n_queries ignored)
                 'shared'   — n_queries shared queries, then linear to n_units
    grid_size  : patch grid side length (14 for ViT-B/14 on 224-px images)
    dropout    : dropout on attention weights
    """

    def __init__(
        self,
        d_feat:    int = 768,
        n_units:   int = 300,
        n_queries: int = 32,
        n_heads:   int = 8,
        mode:      Literal['per_unit', 'shared'] = 'shared',
        grid_size: int = 14,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.d_feat    = d_feat
        self.n_units   = n_units
        self.n_queries = n_queries if mode == 'shared' else n_units
        self.n_heads   = n_heads
        self.mode      = mode
        assert d_feat % n_heads == 0, "d_feat must be divisible by n_heads"

        # Learnable queries
        self.queries = nn.Parameter(
            torch.randn(self.n_queries, d_feat) * 0.02
        )

        # Cross-attention: Q from queries, K/V from patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_feat,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_feat),
            nn.Linear(d_feat, d_feat * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feat * 2, d_feat),
        )

        # Final projection to neuron responses
        if mode == 'shared':
            # (n_queries * d_feat) → n_units
            self.out_proj = nn.Linear(self.n_queries * d_feat, n_units)
        else:
            # per-unit: one linear per query (weight tying via batched linear)
            self.out_proj = nn.Linear(d_feat, 1, bias=True)
            # This is applied independently to each query's output

        # Positional encoding for patch keys (fixed, not learned)
        pos_emb = make_2d_sincos_pos_embed(d_feat, grid_size)  # (N_patches, d_feat)
        self.register_buffer('pos_emb', pos_emb.unsqueeze(0))  # (1, N_patches, d_feat)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.queries)
        if isinstance(self.out_proj, nn.Linear):
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_tokens : (B, N_patches, d_feat)  — DINOv2 patch tokens

        Returns
        -------
        pred : (B, n_units)
        """
        B = patch_tokens.shape[0]

        # Add positional encoding to keys
        keys = patch_tokens + self.pos_emb   # (B, N_patches, d_feat)

        # Expand queries across batch
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, n_queries, d_feat)

        # Cross-attention: queries attend to patch tokens
        attended, _ = self.cross_attn(Q, keys, keys)       # (B, n_queries, d_feat)

        # FFN residual
        attended = attended + self.ffn(attended)            # (B, n_queries, d_feat)

        # Project to responses
        if self.mode == 'shared':
            flat = attended.reshape(B, -1)                 # (B, n_queries * d_feat)
            pred = self.out_proj(flat)                     # (B, n_units)
        else:
            # per_unit: each query → its own neuron's response
            pred = self.out_proj(attended).squeeze(-1)     # (B, n_units)

        return pred


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_tben(
    model:       TBEnReadout,
    features:    np.ndarray,        # (N_images, N_patches, d_feat)
    responses:   np.ndarray,        # (N_images, N_units)
    n_epochs:    int   = 100,
    lr:          float = 1e-3,
    weight_decay:float = 1e-4,
    batch_size:  int   = 64,
    test_size:   float = 0.2,
    device:      str   = 'cpu',
    verbose:     bool  = True,
) -> dict:
    """
    Train TBEn readout and return per-unit test R².

    Returns
    -------
    dict with keys:
        'test_r2'      : (N_units,) float32
        'train_r2'     : (N_units,) float32
        'loss_curve'   : list of epoch train losses
        'model'        : trained TBEnReadout
    """
    model = model.to(device)

    # Train/test split
    idx = np.arange(len(features))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=42)

    X_all = torch.tensor(features,   dtype=torch.float32)
    Y_all = torch.tensor(responses,  dtype=torch.float32)

    X_train, Y_train = X_all[train_idx].to(device), Y_all[train_idx].to(device)
    X_test,  Y_test  = X_all[test_idx].to(device),  Y_all[test_idx].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loss_curve = []
    model.train()

    for epoch in range(n_epochs):
        perm   = torch.randperm(len(X_train), device=device)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_train), batch_size):
            idx_b = perm[start:start + batch_size]
            xb, yb = X_train[idx_b], Y_train[idx_b]

            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss  += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = ep_loss / max(n_batches, 1)
        loss_curve.append(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        yhat_test  = model(X_test).cpu().numpy()
        yhat_train = model(X_train).cpu().numpy()

    y_test  = Y_test.cpu().numpy()
    y_train = Y_train.cpu().numpy()

    test_r2  = _r2_per_unit(y_test,  yhat_test)
    train_r2 = _r2_per_unit(y_train, yhat_train)

    return {
        'test_r2':   test_r2,
        'train_r2':  train_r2,
        'loss_curve': loss_curve,
        'model':     model.cpu(),
    }


def _r2_per_unit(y_true: np.ndarray, y_pred: np.ndarray,
                 min_var: float = 1e-6) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > min_var, 1.0 - ss_res / ss_tot, np.nan)
    return np.clip(r2, -1.0, 1.0).astype(np.float32)
