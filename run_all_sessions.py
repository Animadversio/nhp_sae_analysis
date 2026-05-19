"""
run_all_sessions.py
===================
Run all 3 analysis tasks across all 59 GoodUnit sessions.
DINOv2 features are shared (pre-cached). Neural data is loaded per session.
Results are saved to results/{session_name}/.

Settings: dict_size=2048, n_steps=4000 (best from sweep), n_epochs=200 for TBEn.

Usage:
    python run_all_sessions.py --device cuda [--tasks 1 2 3] [--skip_existing]
"""
import argparse, os, pickle, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis')

parser = argparse.ArgumentParser()
parser.add_argument('--device',        default='cuda')
parser.add_argument('--tasks',         nargs='+', type=int, default=[1, 2, 3])
parser.add_argument('--skip_existing', action='store_true',
                    help='Skip sessions whose results dir already has all_results.pkl')
parser.add_argument('--dict_size',     type=int, default=2048)
parser.add_argument('--sae_steps',     type=int, default=4000)
parser.add_argument('--tben_epochs',   type=int, default=200)
args = parser.parse_args()

DATA_DIR   = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3')
WORK_DIR   = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis')
FEAT_CACHE = WORK_DIR / 'cache' / 'dinov2_spatial_block11.pkl'

assert FEAT_CACHE.exists(), f"Feature cache not found: {FEAT_CACHE}\nRun step1 first."

# Load shared DINOv2 features once
print("Loading DINOv2 features (shared across sessions)...")
with open(FEAT_CACHE, 'rb') as f:
    feat_cache = pickle.load(f)
patch_tokens_all = feat_cache['blocks.11_patch_spatial'].astype('float32')  # (N, 196, 768)
cls_tokens_all   = feat_cache['blocks.11_cls'].astype('float32')             # (N, 768)
print(f"  patch_tokens: {patch_tokens_all.shape}  cls: {cls_tokens_all.shape}")

# Collect all GoodUnit mat files
mat_files = sorted(DATA_DIR.glob('GoodUnit_*.mat'))
print(f"\nFound {len(mat_files)} sessions.")

# Import analysis modules
import h5py
from run_extensions import run_task1, run_task3
from spatial_sae import train_spatial_sae, encode_patches, mean_pool_codes as spatial_pool
from sae_features import fit_and_encode, mean_pool_codes
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA


def _ridge_r2(X, responses, train_idx, time_indices, label):
    N = X.shape[0]
    test_idx = np.setdiff1d(np.arange(N), train_idx)
    n_t = len(time_indices)
    n_u = responses.shape[0]
    r2  = np.full((n_t, n_u), np.nan, np.float32)
    pca = PCA(n_components=min(256, X.shape[1], len(train_idx)))
    Xtr = pca.fit_transform(X[train_idx])
    Xte = pca.transform(X[test_idx])
    for ti, tidx in enumerate(time_indices):
        y   = responses[:, tidx, :].T
        clf = RidgeCV(alphas=np.logspace(-2, 6, 20), alpha_per_target=True)
        clf.fit(Xtr, y[train_idx])
        yhat = clf.predict(Xte)
        ss_res = ((y[test_idx] - yhat) ** 2).sum(0)
        ss_tot = ((y[test_idx] - y[test_idx].mean(0)) ** 2).sum(0)
        r2[ti] = np.where(ss_tot > 1e-6, 1 - ss_res / ss_tot, np.nan)
    peak = float(np.nanmax(np.nanmean(r2, axis=1)))
    print(f"    [{label}] peak mean R² = {peak:.4f}")
    return r2


def run_task2_spatial(data, device, sae_steps, dict_size):
    patch_tokens = data['patch_tokens']
    responses    = data['responses']
    train_idx    = data['train_idx']
    time_indices = data['time_indices']

    raw_feat = patch_tokens.mean(axis=1)
    r2_raw = _ridge_r2(raw_feat, responses, train_idx, time_indices, 'Raw')

    print(f"    Training SAE (dict={dict_size})...")
    sae_codes, _, _ = fit_and_encode('SAE', patch_tokens, d_dict=dict_size, k=64,
                                     n_epochs=max(20, sae_steps // 50),
                                     device=device, verbose=False)
    r2_sae = _ridge_r2(mean_pool_codes(sae_codes), responses, train_idx,
                        time_indices, 'SAE')

    print(f"    Training Spatial SAE (dict={dict_size}, steps={sae_steps})...")
    ae = train_spatial_sae(patch_tokens, n_steps=sae_steps, dict_size=dict_size,
                           k=64, device=device, verbose=False)
    codes   = encode_patches(ae, patch_tokens, device=device)
    r2_sae2 = _ridge_r2(spatial_pool(codes), responses, train_idx,
                         time_indices, 'Spatial SAE')

    return {'raw': r2_raw, 'sae': r2_sae, 'spatial_sae': r2_sae2}


def load_neural_session(mat_path, n_feat_images):
    try:
        with h5py.File(mat_path, 'r') as f:
            refs = f['GoodUnitStrc']['response_matrix_img']
            resp = np.stack([f[refs[i, 0]][()] for i in range(refs.shape[0])], axis=0)
    except Exception as e:
        print(f"  ERROR loading {mat_path.name}: {e}")
        return None

    n_use = min(resp.shape[2], n_feat_images)
    resp  = resp[:, :, :n_use].astype('float32')
    n_units, n_time, n_images = resp.shape
    rng = np.random.default_rng(42)
    train_idx = rng.permutation(n_images)[:int(n_images * 0.8)]

    # Restrict to evoked response window: 50–300 ms post-onset
    # PSTH: -49 to 400 ms, 450 bins → index i corresponds to (-49 + i) ms
    # 50 ms → index 99,  300 ms → index 349
    evoked_indices = np.arange(99, 350)
    t_axis_full    = np.linspace(-49, 400, n_time)

    return {
        'responses':    resp,
        'train_idx':    train_idx,
        'time_indices': evoked_indices,
        't_axis':       t_axis_full,
        'n_images':     n_images,
        'n_units':      n_units,
    }


# ── Main loop ──────────────────────────────────────────────────────────────
summary_rows = []
for idx, mat_path in enumerate(mat_files):
    session = mat_path.stem
    out_dir = WORK_DIR / 'results' / session
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and (out_dir / 'all_results.pkl').exists():
        print(f"\n[{idx+1}/{len(mat_files)}] SKIP (exists): {session}")
        continue

    print(f"\n{'='*70}")
    print(f"[{idx+1}/{len(mat_files)}] {session}")
    t0 = time.time()

    neural = load_neural_session(mat_path, patch_tokens_all.shape[0])
    if neural is None:
        continue

    n_images     = neural['n_images']
    patch_tokens = patch_tokens_all[:n_images]
    cls_tokens   = cls_tokens_all[:n_images]

    data = {
        'patch_tokens': patch_tokens,
        'cls_tokens':   cls_tokens,
        'responses':    neural['responses'],
        'train_idx':    neural['train_idx'],
        'time_indices': neural['time_indices'],
        't_axis':       neural['t_axis'],
        'n_images':     n_images,
        'd_feat':       patch_tokens.shape[-1],
        'n_units':      neural['n_units'],
    }
    print(f"  n_units={neural['n_units']}, n_images={n_images}")

    all_results = {}

    if 1 in args.tasks:
        print("  Task 1: TBEn vs Ridge...")
        r = run_task1(data, device=args.device, n_epochs=args.tben_epochs)
        all_results['task1'] = r
        np.save(out_dir / 'task1_ridge_r2.npy', r['ridge_r2'])
        np.save(out_dir / 'task1_tben_r2.npy',  r['tben_r2'])

    if 2 in args.tasks:
        print("  Task 2: Raw vs SAE vs Spatial SAE...")
        r = run_task2_spatial(data, device=args.device,
                              sae_steps=args.sae_steps,
                              dict_size=args.dict_size)
        all_results['task2'] = r
        for name, arr in r.items():
            np.save(out_dir / f'task2_{name}_r2.npy', arr)

    if 3 in args.tasks:
        print("  Task 3: Token aggregation...")
        try:
            r = run_task3(data, device=args.device,
                          n_time_bins=len(neural['time_indices']))
            all_results['task3'] = r
            for name, d in r.items():
                np.save(out_dir / f'task3_{name}_r2.npy', d['r2'])
        except Exception as e:
            print(f"  Task 3 error (skipping): {e}")

    with open(out_dir / 'all_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    elapsed = time.time() - t0
    row = {'session': session, 'n_units': neural['n_units'],
           'n_images': n_images, 'elapsed_min': round(elapsed / 60, 1)}
    if 1 in args.tasks:
        row['t1_ridge'] = round(float(np.nanmean(all_results['task1']['ridge_r2'])), 4)
        row['t1_tben']  = round(float(np.nanmean(all_results['task1']['tben_r2'])),  4)
    if 2 in args.tasks:
        for k, v in all_results['task2'].items():
            row[f't2_{k}'] = round(float(np.nanmax(np.nanmean(v, axis=1))), 4)
    if 3 in args.tasks and 'task3' in all_results:
        for k, v in all_results['task3'].items():
            row[f't3_{k}'] = round(float(np.nanmax(np.nanmean(v['r2'], axis=1))), 4)
    summary_rows.append(row)
    print(f"  Done in {elapsed/60:.1f} min")

    # Flush running summary after each session
    import pandas as pd
    pd.DataFrame(summary_rows).to_csv(
        WORK_DIR / 'results' / 'all_sessions_summary.csv', index=False)

# Final summary
import pandas as pd
if summary_rows:
    df = pd.DataFrame(summary_rows)
    df.to_csv(WORK_DIR / 'results' / 'all_sessions_summary.csv', index=False)
    print(f"\n\n{'='*70}")
    print("ALL SESSIONS COMPLETE")
    print(df.to_string(index=False))
    print(f"\nSaved -> results/all_sessions_summary.csv")
else:
    print("\nNo sessions processed.")
