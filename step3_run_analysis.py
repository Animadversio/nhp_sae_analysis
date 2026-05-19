"""
Step 3: 跑三个分析任务
运行方式: python step3_run_analysis.py --tasks 1 2 3 --device cuda --n_epochs 100
"""
import argparse, os, pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--tasks',    nargs='+', type=int, default=[1, 2, 3])
parser.add_argument('--device',   default='cuda')
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--block',    type=int, default=11)
args = parser.parse_args()

FEAT_CACHE   = f'cache/dinov2_spatial_block{args.block}.pkl'
NEURAL_CACHE = 'cache/neural_data.pkl'
RESULTS_DIR  = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

assert os.path.exists(FEAT_CACHE),   f"Not found: {FEAT_CACHE}\nRun step1 first."
assert os.path.exists(NEURAL_CACHE), f"Not found: {NEURAL_CACHE}\nRun step2 first."

print("Loading caches...")
with open(FEAT_CACHE, 'rb') as f:
    feat_cache = pickle.load(f)
with open(NEURAL_CACHE, 'rb') as f:
    neural = pickle.load(f)

patch_tokens = feat_cache[f'blocks.{args.block}_patch_spatial'][:neural['n_images']]
cls_tokens   = feat_cache[f'blocks.{args.block}_cls'][:neural['n_images']]

data = {
    'patch_tokens': patch_tokens,
    'cls_tokens':   cls_tokens,
    'responses':    neural['responses'],
    'train_idx':    neural['train_idx'],
    'time_indices': neural['time_indices'],
    't_axis':       neural['t_axis'],
    'n_images':     neural['n_images'],
    'd_feat':       patch_tokens.shape[-1],
    'n_units':      neural['n_units'],
}

print(f"Images: {data['n_images']},  Units: {data['n_units']},  "
      f"patch_tokens: {patch_tokens.shape},  Device: {args.device}")

from run_extensions import run_task1, run_task2, run_task3
all_results = {}

if 1 in args.tasks:
    print("\n--- Task 1: TBEn vs Ridge ---")
    r = run_task1(data, device=args.device, n_epochs=args.n_epochs)
    all_results['task1'] = r
    np.save(f'{RESULTS_DIR}/task1_ridge_r2.npy', r['ridge_r2'])
    np.save(f'{RESULTS_DIR}/task1_tben_r2.npy',  r['tben_r2'])
    print(f"Saved -> {RESULTS_DIR}/task1_*.npy")

if 2 in args.tasks:
    print("\n--- Task 2: Raw vs SAE vs Spatial SAE ---")
    r = run_task2(data, device=args.device, sae_epochs=args.n_epochs,
                  n_time_bins=len(data['time_indices']))
    all_results['task2'] = r
    for name, arr in r.items():
        np.save(f'{RESULTS_DIR}/task2_{name}_r2.npy', arr)
    print(f"Saved -> {RESULTS_DIR}/task2_*.npy")

if 3 in args.tasks:
    print("\n--- Task 3: Token aggregation methods ---")
    r = run_task3(data, device=args.device,
                  n_time_bins=len(data['time_indices']))
    all_results['task3'] = r
    for name, arr in r.items():
        np.save(f'{RESULTS_DIR}/task3_{name}_r2.npy', arr['r2'])
    print(f"Saved -> {RESULTS_DIR}/task3_*.npy")

with open(f'{RESULTS_DIR}/all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print(f"\n✓ Done. Results -> {RESULTS_DIR}/")
