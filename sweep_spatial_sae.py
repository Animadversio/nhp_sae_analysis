"""
sweep_spatial_sae.py
====================
Grid search over dict_size and n_steps for spatial SAE.
Uses the pre-extracted DINOv2 features and pre-loaded neural data.

Usage:
    python sweep_spatial_sae.py --device cuda
"""
import argparse, pickle, sys
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

sys.path.insert(0, '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis')
from spatial_sae import train_spatial_sae, encode_patches, mean_pool_codes

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

FEAT_CACHE   = 'cache/dinov2_spatial_block11.pkl'
NEURAL_CACHE = 'cache/neural_data.pkl'
RESULTS_DIR  = 'results'

with open(FEAT_CACHE, 'rb') as f:
    feat_cache = pickle.load(f)
with open(NEURAL_CACHE, 'rb') as f:
    neural = pickle.load(f)

patch_tokens = feat_cache['blocks.11_patch_spatial'][:neural['n_images']]  # (N, 196, 768)
responses    = neural['responses']   # (n_units, n_time, n_images)
train_idx    = neural['train_idx']
N            = patch_tokens.shape[0]
test_idx     = np.setdiff1d(np.arange(N), train_idx)

# Pick 5 representative time bins covering the response window (50-250ms)
# PSTH: -49 to 400ms, 450 bins → bin for t ms = t + 49
time_bins = [99, 124, 149, 174, 199]  # 50, 75, 100, 125, 150 ms post-onset

def ridge_r2_peak(features: np.ndarray, label: str) -> float:
    """Run Ridge at each time bin, return peak mean R² across units."""
    n_u = responses.shape[0]
    n_t = len(time_bins)
    r2  = np.full((n_t, n_u), np.nan, np.float32)

    pca = PCA(n_components=min(256, features.shape[1], len(train_idx)))
    Xtr = pca.fit_transform(features[train_idx])
    Xte = pca.transform(features[test_idx])

    for ti, tidx in enumerate(time_bins):
        y   = responses[:, tidx, :].T          # (N, n_units)
        clf = RidgeCV(alphas=np.logspace(-2, 6, 20), alpha_per_target=True)
        clf.fit(Xtr, y[train_idx])
        yhat = clf.predict(Xte)
        ss_res = ((y[test_idx] - yhat) ** 2).sum(0)
        ss_tot = ((y[test_idx] - y[test_idx].mean(0)) ** 2).sum(0)
        r2[ti] = np.where(ss_tot > 1e-6, 1 - ss_res / ss_tot, np.nan)

    peak = float(np.nanmax(r2.mean(axis=1)))
    print(f"  {label}: peak mean R² = {peak:.4f}")
    return peak

# Baseline: raw mean-pool
raw_features = patch_tokens.mean(axis=1)
print("=== Baseline ===")
baseline_r2 = ridge_r2_peak(raw_features, 'Raw mean-pool')

# Grid
dict_sizes = [256, 512, 1024, 2048]
n_steps_list = [500, 1500, 4000]

results_grid = {}
print("\n=== Spatial SAE grid search ===")
for dict_size in dict_sizes:
    for n_steps in n_steps_list:
        label = f"dict={dict_size} steps={n_steps}"
        print(f"\n--- {label} ---")
        ae = train_spatial_sae(
            patch_tokens=patch_tokens,
            n_steps=n_steps,
            dict_size=dict_size,
            k=max(16, dict_size // 32),
            device=args.device,
            verbose=False,
        )
        codes    = encode_patches(ae, patch_tokens, device=args.device)
        features = mean_pool_codes(codes)
        r2       = ridge_r2_peak(features, label)
        results_grid[(dict_size, n_steps)] = r2

print("\n\n=== SWEEP SUMMARY ===")
print(f"{'dict_size':>10} {'n_steps':>8} {'peak R²':>10}  vs baseline ({baseline_r2:.4f})")
for (ds, ns), r2 in sorted(results_grid.items()):
    delta = r2 - baseline_r2
    print(f"{ds:>10} {ns:>8} {r2:>10.4f}  ({delta:+.4f})")

# Save
np.save(f'{RESULTS_DIR}/sweep_spatial_sae.npy',
        np.array(list(results_grid.values()), dtype=np.float32))
with open(f'{RESULTS_DIR}/sweep_spatial_sae_keys.txt', 'w') as f:
    for k, v in results_grid.items():
        f.write(f"{k[0]},{k[1]},{v:.6f}\n")
print("\nSaved sweep results.")
