"""
Token Aggregation Methods for Neural Encoding
=============================================
Task 3: Systematic comparison of how to aggregate patch tokens
before (or instead of) the readout step.

Methods compared
----------------
1. mean_pool      — average all patch tokens  (baseline, equivalent to
                    "let every token predict the same response")
2. max_pool       — element-wise max over patch tokens
3. cls_token      — just the CLS token from DINOv2
4. concat_regress — concatenate all tokens, then Ridge regress
                    (high capacity but needs PCA first due to dimensionality)
5. attn_pool      — learned single-head attention pooling
                    (a lightweight version of TBEn with one global query)
6. weighted_mean  — saliency / gradient-based spatial weighting

Each method returns (N_images, D_out) features that can then be passed
to the standard Ridge regression pipeline in core/regression.py.

Usage
-----
    from token_aggregation import compare_aggregation_methods
    from core.regression import fit_pca, time_resolved_regression

    results = compare_aggregation_methods(
        patch_tokens = features,   # (N, P, D)
        cls_tokens   = cls_feats,  # (N, D)
        responses    = resp,       # (N_units, N_time, N_images)
        train_idx    = train_idx,
        time_indices = time_idx,
    )
    # results: dict[method_name → {'r2': (T, U), 'r2_train': (T, U)}]
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from typing import Optional

ALPHAS   = np.logspace(-2, 6, 25)
N_PCA    = 200
MIN_VAR  = 1e-6


# ---------------------------------------------------------------------------
# Aggregation functions (numpy, no grad)
# ---------------------------------------------------------------------------

def mean_pool(patch_tokens: np.ndarray) -> np.ndarray:
    """(N, P, D) → (N, D)  average over patches."""
    return patch_tokens.mean(axis=1)


def max_pool(patch_tokens: np.ndarray) -> np.ndarray:
    """(N, P, D) → (N, D)  element-wise max over patches."""
    return patch_tokens.max(axis=1)


def cls_token(patch_tokens: np.ndarray,
              cls_feats: np.ndarray) -> np.ndarray:
    """Return pre-extracted CLS features.  (N, D)"""
    return cls_feats


def concat_then_pca(patch_tokens: np.ndarray,
                    train_idx: np.ndarray,
                    n_components: int = N_PCA) -> tuple[np.ndarray, PCA]:
    """
    Flatten patch tokens → (N, P*D), fit PCA on train split.
    Returns (N, n_components) and the fitted PCA.
    """
    N, P, D = patch_tokens.shape
    flat    = patch_tokens.reshape(N, P * D)
    pca     = PCA(n_components=min(n_components, flat.shape[1], len(train_idx)))
    pca.fit(flat[train_idx])
    return pca.transform(flat), pca


# ---------------------------------------------------------------------------
# Learned attention pooling (lightweight, single-query cross-attention)
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Single learnable query attends to all patch tokens → pooled (D,) vector.
    Much lighter than full TBEn — one global query, no positional encoding.
    Suitable as a feature aggregator before Ridge regression.
    """

    def __init__(self, d_feat: int = 768, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_feat) * 0.02)
        self.attn  = nn.MultiheadAttention(d_feat, n_heads,
                                            batch_first=True, dropout=0.0)
        self.norm  = nn.LayerNorm(d_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, P, D) → (B, D)"""
        B  = x.shape[0]
        q  = self.query.expand(B, -1, -1)          # (B, 1, D)
        out, _ = self.attn(q, x, x)                # (B, 1, D)
        return self.norm(out.squeeze(1))            # (B, D)


def fit_attn_pool(
    patch_tokens: np.ndarray,    # (N, P, D)
    train_idx:    np.ndarray,
    n_epochs:     int   = 30,
    lr:           float = 1e-3,
    device:       str   = 'cpu',
    verbose:      bool  = False,
) -> tuple[np.ndarray, AttentionPooling]:
    """
    Train attention pooling to minimise reconstruction loss (auto-encoder
    objective on pooled representation → mean).  Returns pooled features (N, D).

    Note: We train with a simple proxy objective (predict the mean token)
    so no neural data is needed.  Alternatively you can train jointly with
    the neural loss — see compare_aggregation_methods for that path.
    """
    X   = torch.tensor(patch_tokens, dtype=torch.float32)
    tgt = X.mean(dim=1)           # proxy target: mean token

    model = AttentionPooling(d_feat=patch_tokens.shape[-1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr  = X[train_idx].to(device)
    tgt_tr = tgt[train_idx].to(device)

    model.train()
    for ep in range(n_epochs):
        opt.zero_grad()
        out  = model(X_tr)
        loss = F.mse_loss(out, tgt_tr)
        loss.backward()
        opt.step()
        if verbose and ep % 10 == 0:
            print(f"  AttnPool epoch {ep}/{n_epochs}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pooled = model(X.to(device)).cpu().numpy()   # (N, D)

    return pooled, model.cpu()


# ---------------------------------------------------------------------------
# Ridge regression helpers (mirrors core/regression.py)
# ---------------------------------------------------------------------------

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2     = np.where(ss_tot > MIN_VAR, 1 - ss_res / ss_tot, np.nan)
    return np.clip(r2, -1, 1).astype(np.float32)


def _fit_pca_split(X: np.ndarray, train_idx: np.ndarray,
                   n: int = N_PCA) -> tuple[np.ndarray, np.ndarray]:
    test_idx = np.setdiff1d(np.arange(len(X)), train_idx)
    n_comp = min(n, X.shape[1], len(train_idx))
    pca = PCA(n_components=n_comp)
    Xtr = pca.fit_transform(X[train_idx])
    Xte = pca.transform(X[test_idx])
    return Xtr, Xte


def _time_resolved_ridge(
    X_train:     np.ndarray,   # (n_train, d)
    X_test:      np.ndarray,   # (n_test,  d)
    response:    np.ndarray,   # (n_units, n_time, n_images)
    train_idx:   np.ndarray,
    time_indices: np.ndarray,
) -> dict:
    """Run per-time-bin multi-output Ridge and return R² matrices."""
    test_idx = np.setdiff1d(np.arange(response.shape[2]), train_idx)
    n_t      = len(time_indices)
    n_units  = response.shape[0]

    r2       = np.full((n_t, n_units), np.nan, np.float32)
    r2_train = np.full((n_t, n_units), np.nan, np.float32)

    for ti, tidx in enumerate(time_indices):
        y = response[:, tidx, :].T                  # (n_images, n_units)
        clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        clf.fit(X_train, y[train_idx])
        r2[ti]       = _safe_r2(y[test_idx],  clf.predict(X_test))
        r2_train[ti] = _safe_r2(y[train_idx], clf.predict(X_train))

    return {'r2': r2, 'r2_train': r2_train}


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_aggregation_methods(
    patch_tokens:  np.ndarray,           # (N_images, N_patches, D_feat)
    cls_feats:     np.ndarray,           # (N_images, D_feat)
    responses:     np.ndarray,           # (N_units, N_time, N_images)
    train_idx:     np.ndarray,
    time_indices:  np.ndarray,
    n_pca:         int  = N_PCA,
    device:        str  = 'cpu',
    verbose:       bool = True,
    methods:       Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Compare all token aggregation strategies using time-resolved Ridge.

    Parameters
    ----------
    patch_tokens  : (N, P, D) — DINOv2 patch tokens (excluding CLS/registers)
    cls_feats     : (N, D)    — pre-extracted CLS token
    responses     : (N_units, N_time, N_images) — neural firing rates
    train_idx     : indices of training images
    time_indices  : which time bins to run regression on
    methods       : subset of methods to run; None → run all

    Returns
    -------
    dict  method_name → {'r2': (T, U), 'r2_train': (T, U)}
    """
    ALL_METHODS = ['mean_pool', 'max_pool', 'cls_token',
                   'concat_regress', 'attn_pool']
    if methods is None:
        methods = ALL_METHODS

    results = {}

    def _run(name: str, X: np.ndarray):
        if verbose:
            print(f"\n[{name}] feature shape: {X.shape}")
        Xtr, Xte = _fit_pca_split(X, train_idx, n=n_pca)
        r        = _time_resolved_ridge(Xtr, Xte, responses, train_idx, time_indices)
        results[name] = r
        if verbose:
            peak_r2 = np.nanmax(r['r2'].mean(axis=1))
            print(f"[{name}] peak mean R²={peak_r2:.4f}")

    # 1. Mean pool
    if 'mean_pool' in methods:
        _run('mean_pool', mean_pool(patch_tokens))

    # 2. Max pool
    if 'max_pool' in methods:
        _run('max_pool', max_pool(patch_tokens))

    # 3. CLS token
    if 'cls_token' in methods:
        _run('cls_token', cls_feats)

    # 4. Concat + PCA (handles the huge P*D dimensionality)
    if 'concat_regress' in methods:
        if verbose:
            print("\n[concat_regress] fitting PCA on flattened tokens...")
        X_concat, _ = concat_then_pca(patch_tokens, train_idx, n_components=n_pca)
        _run('concat_regress', X_concat)

    # 5. Attention pooling (lightweight learned aggregation)
    if 'attn_pool' in methods:
        if verbose:
            print("\n[attn_pool] training attention pooling...")
        pooled, _ = fit_attn_pool(patch_tokens, train_idx,
                                   device=device, verbose=verbose)
        _run('attn_pool', pooled)

    return results


# ---------------------------------------------------------------------------
# Summary / plotting helper
# ---------------------------------------------------------------------------

def summarise_aggregation_results(
    results:      dict[str, dict],
    time_axis:    np.ndarray,
    method_order: Optional[list[str]] = None,
) -> None:
    """
    Print a summary table and optionally plot R² curves.

    Parameters
    ----------
    results    : output of compare_aggregation_methods
    time_axis  : (T,) array of time points in ms (for display)
    """
    if method_order is None:
        method_order = list(results.keys())

    print("\n" + "=" * 60)
    print(f"{'Method':<20} {'Peak mean R²':>14} {'Time of peak':>14}")
    print("=" * 60)

    for name in method_order:
        if name not in results:
            continue
        r2_t = results[name]['r2'].mean(axis=1)    # (T,) mean over units
        peak  = np.nanmax(r2_t)
        tidx  = np.nanargmax(r2_t)
        tpeak = time_axis[tidx] if tidx < len(time_axis) else tidx
        print(f"  {name:<18} {peak:>14.4f} {tpeak:>12.1f} ms")

    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for name in method_order:
            if name not in results:
                continue
            r2_t = results[name]['r2'].mean(axis=1)
            ax.plot(time_axis, r2_t, label=name)
        ax.axvline(0, color='k', lw=0.8, ls='--', label='stimulus onset')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Mean R² across units')
        ax.set_title('Token aggregation method comparison')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('aggregation_comparison.png', dpi=150)
        print("\nFigure saved → aggregation_comparison.png")
    except ImportError:
        print("(matplotlib not available — skipping plot)")
