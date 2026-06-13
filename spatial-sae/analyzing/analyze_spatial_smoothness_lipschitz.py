"""
analyze_spatial_smoothness_lipschitz.py

Computes spatial smoothness metrics following T-SAE Appendix C.1,
adapted to the spatial domain using TRUE 8-NEIGHBORHOOD patch pairs
instead of row-major sequential pairs.

Key change from original version:
  Row-major scan (sequential) → 8-neighborhood pairs (true spatial adjacency)
  For Fourier/Wavelet/Multiscale: signals are built by sampling random
  spatial walk paths through the patch grid using 8-neighborhood steps.
  For Lipschitz: directly computed over all valid 8-neighbor pairs.

Outputs TWO figures:
  Figure 1: smoothness_metrics_comparison.png
    Four subplots (Fourier, Wavelet, Multiscale, Lipschitz-normalized),
    all using 8-neighborhood pairs. Compares DINOv2_raw, SAE_Full,
    SAE_G0, SAE_G1, SAE_G2, SAE_G3.

  Figure 2: lipschitz_comparison.png
    Two subplots:
      Left:  Lipschitz with token normalization (|Δf| / ||Δx||)
      Right: Lipschitz without normalization (|Δf| only, as in T-SAE paper)

Usage:
    python -u scripts/analyze_spatial_smoothness_lipschitz.py \\
        --parquet data/imagenet_data/train-*.parquet \\
        --ckpt ckpts_multiscale_v1/ae_final.pt \\
        --n_images 500 \\
        --device cuda \\
        --outdir results_multiscale/smoothness
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))

DINO_REPO = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache/hub/facebookresearch_dinov2_main"

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE


# ── Models ────────────────────────────────────────────────────

class DINOFeatureExtractor:
    def __init__(self, model_name="dinov2_vitb14", device="cuda"):
        self.device = device
        self.model = torch.hub.load(
            DINO_REPO, model_name, source="local", trust_repo=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, tensor):
        feats = self.model.forward_features(tensor.to(self.device))
        return feats["x_norm_patchtokens"].squeeze(0)  # [N, D]


def load_sae(ckpt, activation_dim, dict_size, k, device):
    """
    Load SAE from checkpoint. Returns (ae, group_boundaries).
    Supports both ae_final.pt (direct state_dict) and checkpoint_step_N.pt.
    """
    raw = torch.load(ckpt, map_location="cpu")
    state_dict = raw["ae_state_dict"] if isinstance(raw, dict) and "ae_state_dict" in raw else raw

    n_groups = len(state_dict["group_sizes"])
    group_size = dict_size // n_groups
    sizes = [group_size] * (n_groups - 1)
    sizes.append(dict_size - sum(sizes))

    ae = TemporalMatryoshkaBatchTopKSAE(
        activation_dim=activation_dim, dict_size=dict_size,
        k=k, group_sizes=sizes, temporal=True,
    ).to(device)

    raw = torch.load(ckpt, map_location=device)
    state_dict = raw["ae_state_dict"] if isinstance(raw, dict) and "ae_state_dict" in raw else raw
    ae.load_state_dict(state_dict)
    ae.eval()
    # L0 sanity check: verify per-token sparsity matches k
    with torch.no_grad():
        _dummy = torch.randn(256, ae.W_enc.shape[0], device=next(ae.parameters()).device)
        _f  = ae.encode(_dummy, use_threshold=False)
        _l0 = (_f > 0).float().sum(1).mean().item()
        _k  = int(ae.k.item())
    print(f"[L0 check] mean L0={_l0:.1f}  k={_k}  threshold={ae.threshold.item():.4f}")

    boundaries = [0]
    for s in sizes:
        boundaries.append(boundaries[-1] + s)
    print(f"[SAE] Loaded {ckpt}  groups={sizes}  boundaries={boundaries}")
    return ae, boundaries


def get_img_bytes(row):
    d = row["image"]
    return d["bytes"] if isinstance(d, dict) else d

def load_tensor(img_bytes, image_size):
    t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return t(Image.open(BytesIO(img_bytes)).convert("RGB")).unsqueeze(0)


# ── 8-neighborhood utilities ──────────────────────────────────

def get_8neighbors(side, r, c):
    """Return all valid 8-neighborhood positions of (r, c) in a side×side grid."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < side and 0 <= cc < side:
                neighbors.append((rr, cc))
    return neighbors


def sample_spatial_walk(side, n_steps=128, seed=None):
    """
    Sample a random walk path through the patch grid using 8-neighborhood steps.
    Returns list of (r, c) positions of length n_steps.
    """
    if seed is not None:
        random.seed(seed)
    r = random.randrange(side)
    c = random.randrange(side)
    path = [(r, c)]
    for _ in range(n_steps - 1):
        nbrs = get_8neighbors(side, r, c)
        r, c = random.choice(nbrs)
        path.append((r, c))
    return path


def get_all_8neighbor_pairs(side):
    """Return all unique unordered 8-neighbor pairs in a side×side grid."""
    pairs = []
    seen = set()
    for r in range(side):
        for c in range(side):
            for rr, cc in get_8neighbors(side, r, c):
                i = r * side + c
                j = rr * side + cc
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    pairs.append(((r, c), (rr, cc)))
    return pairs


# ── Smoothness metrics (8-neighborhood version) ───────────────

def fourier_smoothness_spatial(grid, n_walks=20, walk_length=128):
    """Fourier smoothness using random spatial walk paths. Lower = smoother."""
    side, _, F = grid.shape
    scores = []
    for walk_idx in range(n_walks):
        path = sample_spatial_walk(side, n_steps=walk_length, seed=walk_idx)
        signal = np.stack([grid[r, c] for r, c in path], axis=0)
        T = signal.shape[0]
        mid = T // 2
        for f in range(F):
            s = signal[:, f]
            if s.max() - s.min() < 1e-8:
                continue
            fft_power = np.abs(np.fft.rfft(s)) ** 2
            low_power  = fft_power[:mid].sum() + 1e-10
            high_power = fft_power[mid:].sum() + 1e-10
            scores.append(high_power / low_power)
    return float(np.mean(scores)) if scores else float("nan")


def wavelet_smoothness_spatial(grid, n_walks=20, walk_length=128, n_levels=3):
    """Wavelet smoothness using random spatial walk paths. Lower = smoother."""
    side, _, F = grid.shape
    scores = []
    for walk_idx in range(n_walks):
        path = sample_spatial_walk(side, n_steps=walk_length, seed=walk_idx)
        signal = np.stack([grid[r, c] for r, c in path], axis=0)
        for f in range(F):
            s = signal[:, f].copy()
            if s.max() - s.min() < 1e-8:
                continue
            total_detail_power = 0.0
            for _ in range(n_levels):
                if len(s) < 2:
                    break
                n = len(s) // 2 * 2
                s = s[:n]
                avg    = (s[0::2] + s[1::2]) / 2.0
                detail = (s[0::2] - s[1::2]) / 2.0
                total_detail_power += (detail ** 2).sum()
                s = avg
            approx_power = (s ** 2).sum() + 1e-10
            scores.append((total_detail_power + 1e-10) / approx_power)
    return float(np.mean(scores)) if scores else float("nan")


def multiscale_smoothness_spatial(grid, n_walks=20, walk_length=128, scales=None):
    """
    Multiscale smoothness (T-SAE evaluation.py):
    ratio = variance(diffs at fine scale=1) / variance(diffs at coarse scale=8)
    Lower = smoother.
    """
    if scales is None:
        scales = [1, 2, 4, 8]
    side, _, F = grid.shape
    scores = []
    for walk_idx in range(n_walks):
        path = sample_spatial_walk(side, n_steps=walk_length, seed=walk_idx)
        signal = np.stack([grid[r, c] for r, c in path], axis=0)  # [T, F]

        active = signal.sum(axis=0) != 0
        signal = signal[:, active]
        if signal.shape[1] == 0:
            continue

        T = signal.shape[0]
        scale_measures = {}
        for scale in scales:
            if scale >= T:
                continue
            diffs = signal[scale:] - signal[:-scale]
            scale_measures[scale] = diffs.var(axis=0).mean()

        valid_scales = [s for s in scales if s < T]
        if len(valid_scales) < 2:
            continue

        fine_scale   = min(valid_scales)
        coarse_scale = max(valid_scales)
        ratio = scale_measures[fine_scale] / (scale_measures[coarse_scale] + 1e-10)
        scores.append(float(ratio))
    return float(np.mean(scores)) if scores else float("nan")


def lipschitz_smoothness_spatial(grid, input_grid, normalized=True):
    """
    Lipschitz smoothness over all true 8-neighbor pairs.
    normalized=True: avg |Δf| / ||Δx||  (lower = smoother)
    normalized=False: avg |Δf|  (lower = smoother)
    """
    side = grid.shape[0]
    F    = grid.shape[2]
    pairs = get_all_8neighbor_pairs(side)

    if normalized:
        input_dists = np.array([
            np.linalg.norm(input_grid[r0, c0] - input_grid[r1, c1])
            for (r0, c0), (r1, c1) in pairs
        ]) + 1e-10
    else:
        input_dists = None

    scores = []
    for f in range(F):
        feat_diffs = np.array([
            abs(float(grid[r0, c0, f]) - float(grid[r1, c1, f]))
            for (r0, c0), (r1, c1) in pairs
        ])
        if feat_diffs.max() < 1e-8:
            continue
        if normalized:
            scores.append((feat_diffs / input_dists).mean())
        else:
            scores.append(feat_diffs.mean())
    return float(np.mean(scores)) if scores else float("nan")


# ── Per-image analysis ────────────────────────────────────────

@torch.no_grad()
def analyze_image(tokens_np, ae, device, group_boundaries, n_walks=20, walk_length=128):
    """
    Compute all smoothness metrics for one image.
    Representations: DINOv2_raw, SAE_Full, SAE_G0 … SAE_G(n-1)
    """
    side = int(math.sqrt(tokens_np.shape[0]))
    grid_raw = tokens_np.reshape(side, side, -1)

    results = {}

    def get_active_mask(grid):
        flat = grid.reshape(-1, grid.shape[-1])
        return (flat > 1e-8).sum(axis=0) >= 1

    def compute_all_metrics(grid, input_grid, name):
        mask = get_active_mask(grid)
        n_total  = grid.shape[-1]
        n_active = int(mask.sum())
        if n_active == 0:
            results[name] = {
                "fourier": float("nan"), "wavelet": float("nan"),
                "multiscale": float("nan"),
                "lipschitz_normalized": float("nan"),
                "lipschitz_raw": float("nan"),
                "n_active_features": 0,
                "pct_active_features": 0.0,
            }
            return
        grid_active = grid[:, :, mask]
        results[name] = {
            "fourier":              fourier_smoothness_spatial(grid_active, n_walks, walk_length),
            "wavelet":              wavelet_smoothness_spatial(grid_active, n_walks, walk_length),
            "multiscale":           multiscale_smoothness_spatial(grid_active, n_walks, walk_length),
            "lipschitz_normalized": lipschitz_smoothness_spatial(grid_active, input_grid, normalized=True),
            "lipschitz_raw":        lipschitz_smoothness_spatial(grid_active, input_grid, normalized=False),
            "n_active_features":    n_active,
            "pct_active_features":  100.0 * n_active / n_total,
        }

    compute_all_metrics(grid_raw, grid_raw, "DINOv2_raw")

    if ae is None:
        return results

    tokens_t = torch.tensor(tokens_np, dtype=torch.float32).to(device)
    features = ae.encode(tokens_t, use_threshold=False).cpu().numpy()
    grid_sae = features.reshape(side, side, -1)

    compute_all_metrics(grid_sae, grid_raw, "SAE_Full")

    n_groups = len(group_boundaries) - 1
    for g in range(n_groups):
        start, end = group_boundaries[g], group_boundaries[g + 1]
        compute_all_metrics(grid_sae[:, :, start:end], grid_raw, f"SAE_G{g}")

    return results


# ── Aggregation ───────────────────────────────────────────────

def aggregate(all_results):
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        for repr_name, metrics in r.items():
            for metric_name, val in metrics.items():
                if not math.isnan(val):
                    acc[repr_name][metric_name].append(val)
    return {
        repr_name: {m: (float(np.mean(v)), float(np.std(v))) for m, v in metrics.items()}
        for repr_name, metrics in acc.items()
    }


# ── Plotting ──────────────────────────────────────────────────

# Color palette shared between figures
COLOR_MAP = {
    "DINOv2_raw": "#2563EB",
    "SAE_Full":   "#374151",
    "SAE_G0":     "#F97316",
    "SAE_G1":     "#7C3AED",
    "SAE_G2":     "#16A34A",
    "SAE_G3":     "#DB2777",
}
GROUP_LABELS = {
    "DINOv2_raw": "Raw",
    "SAE_Full":   "Full",
    "SAE_G0":     "G0\n(dist=4)",
    "SAE_G1":     "G1\n(dist=2)",
    "SAE_G2":     "G2\n(dist=1)",
    "SAE_G3":     "G3\n(recon)",
}


def plot_figure1(agg, outdir):
    """Four subplots: Fourier, Wavelet, Multiscale, Lipschitz-normalized."""
    metrics = ["fourier", "wavelet", "multiscale", "lipschitz_normalized"]
    metric_labels = {
        "fourier":              "Fourier Smoothness\n(high/low freq ratio, lower=smoother)",
        "wavelet":              "Wavelet Smoothness\n(detail/approx ratio, lower=smoother)",
        "multiscale":           "Multiscale Smoothness\n(fine/coarse var ratio, lower=smoother)",
        "lipschitz_normalized": "Lipschitz (normalized)\n(avg |Δf|/||Δx||, lower=smoother)",
    }
    repr_names = list(agg.keys())
    colors = [COLOR_MAP.get(r, "#999999") for r in repr_names]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, metric in zip(axes, metrics):
        means = [agg[r].get(metric, (0, 0))[0] for r in repr_names]
        stds  = [agg[r].get(metric, (0, 0))[1] for r in repr_names]
        x = np.arange(len(repr_names))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS.get(r, r) for r in repr_names], fontsize=9)
        ax.set_title(metric_labels[metric], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        max_std = max(stds) if stds else 0
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_std * 0.05 + 1e-10,
                    f"{mean:.3f}", ha="center", fontsize=7)

    steps_str = GROUP_LABELS.get("_steps_str", "dist=4/2/1/recon")
    plt.suptitle(
        "Spatial Smoothness Metrics (8-neighborhood patch pairs)\n"
        f"Raw vs SAE Full vs Groups G0…G3 (steps={steps_str})",
        fontsize=11,
    )
    plt.tight_layout()
    path = outdir / "smoothness_metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_figure2(agg, outdir):
    """Two subplots: Lipschitz normalized vs raw."""
    repr_names = list(agg.keys())
    colors = [COLOR_MAP.get(r, "#999999") for r in repr_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title, note in zip(
        axes,
        ["lipschitz_normalized", "lipschitz_raw"],
        ["Lipschitz (token-normalized)", "Lipschitz (raw, as in T-SAE paper)"],
        ["avg |Δf| / ||Δx||  —  normalizes by input space distance",
         "avg |Δf|  —  simple absolute feature difference between neighbors"],
    ):
        means = [agg[r].get(metric, (0, 0))[0] for r in repr_names]
        stds  = [agg[r].get(metric, (0, 0))[1] for r in repr_names]
        x = np.arange(len(repr_names))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABELS.get(r, r) for r in repr_names], fontsize=9)
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.set_ylabel("Lipschitz constant (lower = smoother)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        max_std = max(stds) if stds else 0
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_std * 0.05 + 1e-10,
                    f"{mean:.4f}", ha="center", fontsize=8)

    plt.suptitle(
        "Lipschitz Smoothness: Two Formulations\n"
        "Computed over all true 8-neighborhood patch pairs",
        fontsize=11,
    )
    plt.tight_layout()
    path = outdir / "lipschitz_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_table(agg):
    repr_names = list(agg.keys())
    metrics = ["fourier", "wavelet", "multiscale", "lipschitz_normalized", "lipschitz_raw"]
    header = f"{'Representation':<20}" + "".join(f"{m:>24}" for m in metrics) + f"{'active%':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in repr_names:
        row = f"{r:<20}"
        for m in metrics:
            if m in agg.get(r, {}):
                mean, std = agg[r][m]
                row += f"{mean:>14.4f}±{std:<9.4f}"
            else:
                row += f"{'N/A':>24}"
        if "pct_active_features" in agg.get(r, {}):
            pct = agg[r]["pct_active_features"][0]
            row += f"{pct:>9.1f}%"
        print(row)
    print("=" * len(header))


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",     type=str, required=True)
    parser.add_argument("--ckpt",        type=str, default=None)
    parser.add_argument("--n_images",    type=int, default=500)
    parser.add_argument("--n_walks",     type=int, default=20)
    parser.add_argument("--walk_length", type=int, default=128)
    parser.add_argument("--dino_model",  type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--dict_size",   type=int, default=16384)
    parser.add_argument("--k",           type=int, default=32)
    parser.add_argument("--hl_fraction", type=float, default=0.25)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--outdir",      type=str, default="results/smoothness_lipschitz")
    parser.add_argument("--group_steps", type=int, nargs="+", default=None,
                        help="group_steps used in training, e.g. --group_steps 1 2 4 8. "
                             "If provided, overrides the default G0/G1/G2/G3 labels.")
    args = parser.parse_args()

    # Build dynamic GROUP_LABELS if group_steps provided
    if args.group_steps is not None:
        for gi, step in enumerate(args.group_steps):
            key = f"SAE_G{gi}"
            lbl = f"G{gi}\n(step={step})" if step > 0 else f"G{gi}\n(recon)"
            GROUP_LABELS[key] = lbl
        # update suptitle
        steps_str = "/".join(str(s) for s in args.group_steps)
        GROUP_LABELS["_steps_str"] = steps_str

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    print(f"[Data] Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    if len(df) > args.n_images:
        df = df.sample(args.n_images, random_state=42).reset_index(drop=True)
    print(f"[Data] Using {len(df)} images")

    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    ae = None
    group_boundaries = None
    if args.ckpt:
        sample_tensor = load_tensor(get_img_bytes(df.iloc[0]), args.image_size)
        with torch.no_grad():
            activation_dim = extractor.patch_tokens(sample_tensor).shape[-1]
        print(f"[Model] Loading SAE from {args.ckpt}...")
        ae, group_boundaries = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    all_results = []
    print(f"\n[Analyze] Processing {len(df)} images "
          f"({args.n_walks} walks × {args.walk_length} steps each)...")

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 50 == 0:
            print(f"  {idx}/{len(df)}")
        try:
            img_bytes = get_img_bytes(row)
            tensor    = load_tensor(img_bytes, args.image_size)
            with torch.no_grad():
                tokens = extractor.patch_tokens(tensor).cpu().numpy()
            side = int(math.sqrt(tokens.shape[0]))
            if side * side != tokens.shape[0]:
                continue
            result = analyze_image(
                tokens, ae, args.device, group_boundaries,
                n_walks=args.n_walks, walk_length=args.walk_length,
            )
            all_results.append(result)
        except Exception:
            continue

    print(f"[Analyze] Processed {len(all_results)} images")

    agg = aggregate(all_results)
    print_table(agg)
    plot_figure1(agg, outdir)
    plot_figure2(agg, outdir)

    print(f"\n[Done] Results saved to: {outdir}")
    print("  smoothness_metrics_comparison.png  → Figure 1: 4 metrics")
    print("  lipschitz_comparison.png           → Figure 2: normalized vs raw Lipschitz")


if __name__ == "__main__":
    main()
