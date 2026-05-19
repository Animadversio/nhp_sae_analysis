"""
Step 2: 加载神经数据，构建 response matrix
运行方式: python step2_load_neural.py --mat_file "C:\path\to\GoodUnit.mat"
"""
import argparse, os, pickle
import numpy as np, h5py

parser = argparse.ArgumentParser()
parser.add_argument('--mat_file', required=True, help='GoodUnit .mat 文件路径')
parser.add_argument('--block',    type=int, default=11)
args = parser.parse_args()

FEAT_CACHE  = f'cache/dinov2_spatial_block{args.block}.pkl'
NEURAL_CACHE = 'cache/neural_data.pkl'
os.makedirs('cache', exist_ok=True)

assert os.path.exists(FEAT_CACHE), f"Feature cache not found: {FEAT_CACHE}\nRun step1 first."

print(f"Opening: {args.mat_file}")
with h5py.File(args.mat_file, 'r') as f:
    resp_refs = f['GoodUnitStrc']['response_matrix_img']  # (n_units, 1) object refs
    # Each ref points to (n_time, n_images) = (450, 1072); stack → (n_units, n_time, n_images)
    resp = np.stack([f[resp_refs[i, 0]][()] for i in range(resp_refs.shape[0])], axis=0)

print(f"Loaded shape: {resp.shape}  (n_units, n_time, n_images)")

with open(FEAT_CACHE, 'rb') as f:
    feat = pickle.load(f)
n_feat   = feat[list(feat.keys())[0]].shape[0]
n_neural = resp.shape[2]
print(f"Feature images: {n_feat},  Neural images: {n_neural}")

n_use = min(n_feat, n_neural)
resp  = resp[:, :, :n_use]
n_units, n_time, n_images = resp.shape

cache = {
    'responses':    resp.astype('float32'),
    'train_idx':    np.arange(int(n_images * 0.8)),
    'time_indices': np.arange(n_time),
    't_axis':       np.linspace(-50, 400, n_time),
    'n_images':     n_images,
    'n_units':      n_units,
}
with open(NEURAL_CACHE, 'wb') as f:
    pickle.dump(cache, f)
print(f"Saved -> {NEURAL_CACHE}")
print(f"n_units={n_units},  n_time={n_time},  n_images={n_images}")
