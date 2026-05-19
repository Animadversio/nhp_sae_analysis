"""
run_extensions.py
=================
Main entry point for the three new analysis tasks:

  Task 1 — TBEn cross-attention readout (replaces Ridge regression)
  Task 2 — Raw vs SAE vs Spatial SAE features
  Task 3 — Token aggregation method comparison

HOW TO RUN
----------
    python run_extensions.py --data_path /path/to/GoodUnit_*.mat \
                             --tasks 1 2 3 \
                             --device cpu

Minimal smoke-test with synthetic data (no real data needed):
    python run_extensions.py --smoke_test

DEPENDENCIES
------------
    pip install torch numpy scikit-learn scipy matplotlib
    (scipy needed only for loading .mat files)

NOTES FOR INTEGRATION WITH EXISTING CODE
-----------------------------------------
The three modules (tben_readout.py, sae_features.py, token_aggregation.py)
are designed to slot alongside the existing NHP_NSD_analysis codebase.

Input format assumed:
  - DINOv2 patch tokens: (N_images, N_patches, D_feat)
      e.g. the 'blocks.{i}_patch' entries in the existing feature cache,
      but reshaped to keep the spatial (patch) dimension instead of
      mean-pooling.
  - CLS tokens:          (N_images, D_feat)
  - Neural responses:    (N_units, N_time, N_images)  (same as existing code)

The existing code mean-pools patches before regression.  Task 3 compares
this against alternatives.  Tasks 1–2 work on the full (N, P, D) tensor.
"""

import argparse
import sys
import numpy as np

# ── Try importing local extensions ──────────────────────────────────────────
try:
    from tben_readout    import TBEnReadout, train_tben
    from sae_features    import fit_and_encode, mean_pool_codes
    from token_aggregation import (compare_aggregation_methods,
                                   summarise_aggregation_results,
                                   mean_pool)
except ImportError as e:
    print(f"[ERROR] Could not import extension modules: {e}")
    print("Make sure tben_readout.py, sae_features.py and "
          "token_aggregation.py are in the same directory.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic data generator (for smoke-testing without real data)
# ---------------------------------------------------------------------------

def make_synthetic_data(
    n_images:  int = 200,
    n_patches: int = 196,
    d_feat:    int = 768,
    n_units:   int = 50,
    n_time:    int = 18,
    seed:      int = 42,
) -> dict:
    """
    Tiny synthetic dataset that exercises all three tasks.
    Responses are a noisy linear function of mean-pooled features
    so we know Ridge should work reasonably and TBEn should be competitive.
    """
    rng = np.random.default_rng(seed)

    patch_tokens = rng.standard_normal((n_images, n_patches, d_feat)).astype(np.float32)
    cls_tokens   = patch_tokens[:, 0, :].copy()   # just reuse first patch as CLS

    # Random linear ground truth + noise
    W    = rng.standard_normal((d_feat, n_units)).astype(np.float32) * 0.1
    mean = patch_tokens.mean(axis=1)               # (N, D)
    base = (mean @ W)                              # (N, n_units)

    # Temporal modulation: Gaussian bump peaking at t=8
    t_axis = np.linspace(-50, 400, n_time)
    t_mod  = np.exp(-0.5 * ((t_axis - 100) / 40) ** 2).astype(np.float32)

    # response[u, t, i] = t_mod[t] * base[i, u] + noise
    responses = (
        t_mod[None, :, None] *
        base.T[:, None, :]   +
        rng.standard_normal((n_units, n_time, n_images)).astype(np.float32) * 0.3
    )

    train_idx = np.arange(int(n_images * 0.8))
    time_idx  = np.arange(n_time)

    return {
        'patch_tokens': patch_tokens,
        'cls_tokens':   cls_tokens,
        'responses':    responses,
        'train_idx':    train_idx,
        'time_indices': time_idx,
        't_axis':       t_axis,
        'n_images':     n_images,
        'd_feat':       d_feat,
        'n_units':      n_units,
    }


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def run_task1(data: dict, device: str = 'cpu', n_epochs: int = 30):
    """TBEn cross-attention readout vs Ridge baseline."""
    print("\n" + "=" * 60)
    print("TASK 1 — TBEn cross-attention readout")
    print("=" * 60)

    patch_tokens = data['patch_tokens']    # (N, P, D)
    responses    = data['responses']       # (N_u, N_t, N_i)
    train_idx    = data['train_idx']
    N, P, D      = patch_tokens.shape
    n_units      = responses.shape[0]

    # Average response across time as a single prediction target
    # (For a fair comparison to Ridge on the time-resolved task,
    #  you would loop over time bins and call train_tben per bin,
    #  or extend TBEnReadout to output (n_units × n_time).
    #  Here we use the peak-time response as a concise demo.)
    peak_t = responses.mean(axis=2).mean(axis=0).argmax()    # best time bin
    resp_peak = responses[:, peak_t, :].T.astype(np.float32) # (N, n_units)

    print(f"\n  Using peak time bin t={peak_t} for demonstration")
    print(f"  Feature shape : {patch_tokens.shape}")
    print(f"  Response shape: {resp_peak.shape}")

    # ── Ridge baseline (mean-pooled features) ──────────────────────────────
    from sklearn.linear_model import RidgeCV
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    X_mean = patch_tokens.mean(axis=1)     # (N, D)
    test_idx = np.setdiff1d(np.arange(N), train_idx)

    pca = PCA(n_components=min(200, D, len(train_idx)))
    Xtr = pca.fit_transform(X_mean[train_idx])
    Xte = pca.transform(X_mean[test_idx])

    ridge = RidgeCV(alphas=np.logspace(-2, 6, 25), alpha_per_target=True)
    ridge.fit(Xtr, resp_peak[train_idx])
    yhat  = ridge.predict(Xte)
    y_te  = resp_peak[test_idx]
    ss_res = ((y_te - yhat) ** 2).sum(0)
    ss_tot = ((y_te - y_te.mean(0)) ** 2).sum(0)
    r2_ridge = np.where(ss_tot > 1e-6, 1 - ss_res/ss_tot, np.nan)
    print(f"\n  Ridge (mean-pool) — mean test R² = {np.nanmean(r2_ridge):.4f}")

    # ── TBEn (shared-query mode) ───────────────────────────────────────────
    grid_size = int(round(P ** 0.5))   # 14 for 196 patches
    model = TBEnReadout(
        d_feat    = D,
        n_units   = n_units,
        n_queries = min(16, n_units),
        n_heads   = min(4, D // 64),
        mode      = 'shared',
        grid_size = grid_size,
        dropout   = 0.1,
    )
    print(f"\n  Training TBEn (shared, {n_epochs} epochs)...")
    res_tben = train_tben(model, patch_tokens, resp_peak,
                           n_epochs=n_epochs, device=device, verbose=True)
    print(f"\n  TBEn (cross-attn) — mean test R² = {np.nanmean(res_tben['test_r2']):.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n  --- TASK 1 SUMMARY ---")
    print(f"  Ridge (mean-pool):  {np.nanmean(r2_ridge):.4f}")
    print(f"  TBEn (cross-attn):  {np.nanmean(res_tben['test_r2']):.4f}")

    return {'ridge_r2': r2_ridge, 'tben_r2': res_tben['test_r2']}


def run_task2(data: dict, device: str = 'cpu',
              sae_epochs: int = 20, n_time_bins: int = 5):
    """Raw vs SAE vs Spatial SAE features with Ridge readout."""
    print("\n" + "=" * 60)
    print("TASK 2 — Raw vs SAE vs Spatial SAE features")
    print("=" * 60)

    patch_tokens = data['patch_tokens']    # (N, P, D)
    responses    = data['responses']       # (N_u, N_t, N_i)
    train_idx    = data['train_idx']
    time_indices = data['time_indices'][:n_time_bins]  # use subset for speed
    N, P, D      = patch_tokens.shape

    from sklearn.linear_model import RidgeCV
    from sklearn.decomposition import PCA

    def _ridge_r2(X: np.ndarray, method_name: str) -> np.ndarray:
        """Run time-resolved Ridge and return (T, U) R² array."""
        test_idx = np.setdiff1d(np.arange(N), train_idx)
        pca = PCA(n_components=min(200, X.shape[1], len(train_idx)))
        Xtr = pca.fit_transform(X[train_idx])
        Xte = pca.transform(X[test_idx])
        n_t = len(time_indices)
        n_u = responses.shape[0]
        r2  = np.full((n_t, n_u), np.nan, np.float32)
        for ti, tidx in enumerate(time_indices):
            y   = responses[:, tidx, :].T
            clf = RidgeCV(alphas=np.logspace(-2, 6, 25), alpha_per_target=True)
            clf.fit(Xtr, y[train_idx])
            yhat = clf.predict(Xte)
            ss_res = ((y[test_idx] - yhat) ** 2).sum(0)
            ss_tot = ((y[test_idx] - y[test_idx].mean(0)) ** 2).sum(0)
            r2[ti] = np.where(ss_tot > 1e-6, 1 - ss_res/ss_tot, np.nan)
        peak = np.nanmax(r2.mean(axis=1))
        print(f"  [{method_name}] peak mean R² = {peak:.4f}")
        return r2

    # 1. Raw (mean-pooled)
    raw_features = patch_tokens.mean(axis=1)     # (N, D)
    r2_raw = _ridge_r2(raw_features, 'Raw (mean-pool)')

    # 2. Standard SAE (mean-pooled input, original implementation)
    print(f"\n  Training SAE (d_dict=512, {sae_epochs} epochs)...")
    sae_codes, _, _ = fit_and_encode(
        'SAE', patch_tokens, d_dict=512, k=32,
        n_epochs=sae_epochs, device=device, verbose=True
    )
    sae_features = mean_pool_codes(sae_codes)   # (N, d_dict)
    r2_sae = _ridge_r2(sae_features, 'SAE')

    # 3. Spatial SAE (MatryoshkaBatchTopKSAE + spatial contrastive loss)
    from spatial_sae import (train_spatial_sae, encode_patches,
                              mean_pool_codes as spatial_mean_pool)
    n_steps_spatial = max(200, sae_epochs * 10)
    print(f"\n  Training Spatial SAE (dict_size=512, {n_steps_spatial} steps)...")
    spatial_ae = train_spatial_sae(
        patch_tokens=patch_tokens,
        n_steps=n_steps_spatial,
        dict_size=512,
        k=32,
        device=device,
        verbose=True,
    )
    spatial_codes = encode_patches(spatial_ae, patch_tokens, device=device)  # (N, P, 512)
    spatial_features = spatial_mean_pool(spatial_codes)                       # (N, 512)
    r2_spatial = _ridge_r2(spatial_features, 'Spatial SAE')

    print("\n  --- TASK 2 SUMMARY ---")
    for name, r2 in [('Raw        ', r2_raw),
                     ('SAE        ', r2_sae),
                     ('Spatial SAE', r2_spatial)]:
        print(f"  {name}  peak mean R² = {np.nanmax(r2.mean(axis=1)):.4f}")

    return {'raw': r2_raw, 'sae': r2_sae, 'spatial_sae': r2_spatial}


def run_task3(data: dict, device: str = 'cpu', n_time_bins: int = 5):
    """Token aggregation method comparison."""
    print("\n" + "=" * 60)
    print("TASK 3 — Token aggregation method comparison")
    print("=" * 60)

    time_indices = data['time_indices'][:n_time_bins]

    results = compare_aggregation_methods(
        patch_tokens  = data['patch_tokens'],
        cls_feats     = data['cls_tokens'],
        responses     = data['responses'],
        train_idx     = data['train_idx'],
        time_indices  = time_indices,
        device        = device,
        verbose       = True,
    )
    summarise_aggregation_results(results, data['t_axis'][time_indices])
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='NHP_NSD extension analyses')
    p.add_argument('--smoke_test', action='store_true',
                   help='Run on synthetic data (no real data needed)')
    p.add_argument('--tasks', nargs='+', type=int, default=[1, 2, 3],
                   help='Which tasks to run (e.g. --tasks 1 3)')
    p.add_argument('--device', default='cpu',
                   help='torch device: cpu or cuda')
    p.add_argument('--data_path', default=None,
                   help='Path to GoodUnit .mat file (if not smoke_test)')
    p.add_argument('--feature_cache', default=None,
                   help='Path to DINOv2 feature cache .pkl')
    p.add_argument('--n_epochs', type=int, default=30,
                   help='Training epochs for TBEn and SAE')
    p.add_argument('--n_time_bins', type=int, default=5,
                   help='Number of time bins for tasks 2 & 3 (speed)')
    return p.parse_args()


def load_real_data(args) -> dict:
    """
    Load real NHP data.  Returns same dict format as make_synthetic_data.
    Requires:
      - args.data_path  : GoodUnit .mat file
      - args.feature_cache : cached DINOv2 features .pkl (with patch dim)
    """
    import pickle, scipy.io

    print(f"Loading neural data from: {args.data_path}")
    mat  = scipy.io.loadmat(args.data_path)
    # response_matrix_img: (n_units, n_images, n_time) — standard format
    resp = mat['GoodUnitStrc'][0][0]['response_matrix_img']   # adjust field name
    # Reorder to (n_units, n_time, n_images) to match internal convention
    if resp.ndim == 3 and resp.shape[1] != resp.shape[2]:
        resp = resp.transpose(0, 2, 1)

    print(f"Loading DINOv2 features from: {args.feature_cache}")
    with open(args.feature_cache, 'rb') as f:
        feat_cache = pickle.load(f)

    # Use the last block's patch features; assume (N, D) in cache → need (N, P, D)
    # If cached as mean-pooled, note that compare_aggregation is less meaningful.
    # The feature cache ideally has 'blocks.11_patch_spatial': (N, 196, 768)
    key_patch = 'blocks.11_patch_spatial'
    key_cls   = 'blocks.11_cls'
    if key_patch not in feat_cache:
        raise KeyError(
            f"Feature cache must contain '{key_patch}' with shape (N, 196, 768). "
            f"Re-extract features with spatial=True.  Available keys: "
            f"{list(feat_cache.keys())[:10]}"
        )

    patch_tokens = feat_cache[key_patch].astype(np.float32)   # (N, 196, 768)
    cls_tokens   = feat_cache[key_cls].astype(np.float32)     # (N, 768)
    N            = patch_tokens.shape[0]
    train_idx    = np.arange(int(N * 0.8))
    n_time       = resp.shape[1]
    time_axis    = np.linspace(-50, 400, n_time)

    return {
        'patch_tokens': patch_tokens,
        'cls_tokens':   cls_tokens,
        'responses':    resp.astype(np.float32),
        'train_idx':    train_idx,
        'time_indices': np.arange(n_time),
        't_axis':       time_axis,
        'n_images':     N,
        'd_feat':       patch_tokens.shape[-1],
        'n_units':      resp.shape[0],
    }


def main():
    args = parse_args()

    if args.smoke_test or args.data_path is None:
        print("Running smoke test on synthetic data...")
        print("(Pass --data_path and --feature_cache for real data)\n")
        data = make_synthetic_data(n_images=200, n_units=30, n_time=10)
    else:
        data = load_real_data(args)

    print(f"\nData summary:")
    print(f"  Images        : {data['n_images']}")
    print(f"  Units         : {data['n_units']}")
    print(f"  Patch tokens  : {data['patch_tokens'].shape}")
    print(f"  Feature dim   : {data['d_feat']}")
    print(f"  Time bins     : {len(data['time_indices'])}")
    print(f"  Device        : {args.device}")

    all_results = {}

    if 1 in args.tasks:
        all_results['task1'] = run_task1(
            data, device=args.device, n_epochs=args.n_epochs
        )

    if 2 in args.tasks:
        all_results['task2'] = run_task2(
            data, device=args.device,
            sae_epochs=args.n_epochs,
            n_time_bins=args.n_time_bins,
        )

    if 3 in args.tasks:
        all_results['task3'] = run_task3(
            data, device=args.device,
            n_time_bins=args.n_time_bins,
        )

    print("\n\n✓ All requested tasks complete.")
    return all_results


if __name__ == '__main__':
    main()
