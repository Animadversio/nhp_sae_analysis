# NHP SAE Analysis

Sparse Autoencoder (SAE) and Transformer-based encoding analysis of NHP neural responses to NSD images in LOC area.

## Overview

This repo extends the [NHP_NSD_analysis](https://github.com/Animadversio/NHP_NSD_analysis) pipeline with three new analysis tasks:

| Task | Description |
|------|-------------|
| Task 1 | TBEn cross-attention readout vs Ridge regression |
| Task 2 | Raw DINOv2 vs SAE vs Spatial SAE features with Ridge readout |
| Task 3 | Token aggregation comparison (mean/max pool, CLS, flatten+PCA, attn pool) |

## Dataset

- **NSD_N3**: 59 sessions, 5 NHP subjects (JianJian/M1, FaCai/M2, ZhuangZhuang/M3, MaoDan/M4, TuTu/M5)
- LOC area recordings, viewing 1072 NSD natural scene images
- Data path: `/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3/`
- Features: DINOv2 ViT-B/14-reg block 11 patch tokens, cached at `cache/dinov2_spatial_block11.pkl`

## Key Files

| File | Description |
|------|-------------|
| `spatial_sae.py` | Self-contained Spatial SAE — MatryoshkaBatchTopKSAE with spatial contrastive loss |
| `run_all_sessions.py` | Full 59-session pipeline with skip/resume |
| `sweep_spatial_sae.py` | Hyperparameter sweep (dict_size × n_steps) |
| `run_extensions.py` | Task runners for Tasks 1–3 |
| `step1_extract_features.py` | DINOv2 feature extraction and caching |
| `step2_load_neural.py` | Load and preprocess GoodUnit .mat files |
| `step3_run_analysis.py` | Single-session analysis entry point |
| `tben_readout.py` | TBEn cross-attention readout model |
| `sae_features.py` | Standard (non-spatial) SAE feature encoding |
| `token_aggregation.py` | Token aggregation methods (mean/max/cls/attn pool) |

## spatial_sae.py API

```python
from spatial_sae import train_spatial_sae, encode_patches, encode_patches_per_group_topk, mean_pool_codes

# Train
ae = train_spatial_sae(
    patch_tokens,        # (N_images, N_patches, D_feat) float32
    n_steps=4000,        # training steps
    dict_size=2048,      # SAE dictionary size
    k=64,                # global top-k sparsity
    group_fractions=[0.25]*4,  # Matryoshka group fractions
    contrastive_alpha=5.0,     # weight of spatial contrastive loss
    device='cuda',
)

# Encode (global threshold)
codes = encode_patches(ae, patch_tokens)   # (N, P, dict_size)

# Encode (per-group top-k — enforces equal representation from each group)
codes = encode_patches_per_group_topk(ae, patch_tokens, k_per_group=16)

# Aggregate
features = mean_pool_codes(codes)          # (N, dict_size) — for Ridge pipeline
```

## Run full pipeline

```bash
WORK=/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis
PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python

cd $WORK && PYTHONNOUSERSITE=1 $PYTHON run_all_sessions.py \
    --device cuda \
    --tasks 1 2 3 \
    --dict_size 2048 \
    --sae_steps 4000 \
    --tben_epochs 200 \
    --skip_existing
```

Results saved to `results/all_sessions_summary.csv`.

## Key Findings

### Best methods (avg-evoked target, flatten+PCA=128)
| Method | Mean R² (3 sessions) |
|--------|----------------------|
| SAE flatten+PCA | **0.207** |
| SAE max_pool | 0.197 |
| Spatial SAE flatten+PCA | 0.194 |
| Raw DINOv2 flatten+PCA | 0.193 |
| SAE mean_pool | 0.180 |

### TBEn regularization (avg-evoked target)
Best config: `weight_decay=0.01`, `n_epochs=30`, `n_queries=4`, `dropout=0.3`
- Original config (200ep, wd=1e-4): train R²≈1.0, test R²≈-0.23 (severe overfitting)
- Tuned config: train R²≈0.13, test R²≈0.13 (competitive with Ridge)

### Matryoshka group analysis
- Group 2 (features 256–511) most neurally predictive with global threshold encoding
- Group 1 most predictive with per-group top-k encoding
- All groups carry neural information; groups 3–4 capture finer/less-predictive details

### Hyperparameter sweep results
- Best SAE config: `dict_size=2048`, `n_steps≥1500`
- Best PCA dimension for flatten+PCA: 64–128
- TBEn: `weight_decay=0.01`, `n_epochs=30` optimal

## Data format

`GoodUnit_*.mat` files (HDF5 v7.3):
- `response_matrix_img`: (450 time bins × 1072 images) per unit, trial-averaged float32
- PSTH timing: −49 to +400ms at 1ms resolution
- Evoked window: 50–300ms post-onset (indices 99–349)
