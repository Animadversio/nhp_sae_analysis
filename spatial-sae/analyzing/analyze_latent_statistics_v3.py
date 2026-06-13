"""
analyze_latent_statistics_v3.py

Computes per-latent statistics and visualizes:
  - Scatter plot: left=label entropy coloring, right=group membership (G0–G3)
  - Class-specific concept grid (top latents from G0, strongest class)
  - Shared concept grid (top latents from G3, diverse classes)

Usage:
    python scripts/analyze_latent_statistics_v3.py \\
        --parquet data/imagenet_data/train-*.parquet \\
        --ckpt ckpts_multiscale_v1/ae_final.pt \\
        --n_images 2000 \\
        --n_latents 5 \\
        --n_images_per_latent 10 \\
        --device cuda \\
        --outdir results_multiscale/latent_statistics
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
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
        return feats["x_norm_patchtokens"].squeeze(0)

def load_sae(ckpt, activation_dim, dict_size, k, device):
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
    return ae

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

def load_pil(img_bytes, image_size):
    return Image.open(BytesIO(img_bytes)).convert("RGB").resize((image_size, image_size))

def minmax_norm(x, clip_percentile=50):
    """Normalize to [0,1], clipping values below clip_percentile to 0.
    This prevents low-amplitude background activations from appearing colored."""
    x = np.clip(x, 0, None)  # ensure non-negative (TopK output should be >=0)
    if x.max() < 1e-8:
        return np.zeros_like(x)
    thresh = np.percentile(x[x > 0], clip_percentile) if (x > 0).any() else 0
    x = np.where(x >= thresh, x, 0.0)
    hi = x.max()
    if hi < 1e-8:
        return np.zeros_like(x)
    return x / hi


# ── Step 1: Collect statistics ────────────────────────────────

@torch.no_grad()
def collect_latent_statistics(
    df, label_col, extractor, ae, device,
    image_size, dict_size, threshold, top_k_refs,
    encode_batch_size=8,
):
    """
    Returns per-latent stats + per-latent per-class top reference images.
    Encodes encode_batch_size images together per ae.encode() call
    so that BatchTopK competition matches training conditions.
    """
    n_images = len(df)
    activation_count = np.zeros(dict_size, dtype=np.float64)
    activation_sum   = np.zeros(dict_size, dtype=np.float64)
    label_act_sum    = defaultdict(lambda: np.zeros(dict_size, dtype=np.float64))
    label_act_count  = defaultdict(lambda: np.zeros(dict_size, dtype=np.float64))

    top_refs_by_class = [defaultdict(list) for _ in range(dict_size)]
    top_refs_global   = [[] for _ in range(dict_size)]

    rows_list = list(df.iterrows())
    print(f"[Stats] Processing {n_images} images (encode_batch_size={encode_batch_size})...")

    img_idx = 0
    while img_idx < len(rows_list):
        batch_tokens, batch_labels, batch_indices = [], [], []
        for bi in range(encode_batch_size):
            if img_idx + bi >= len(rows_list):
                break
            _, row = rows_list[img_idx + bi]
            try:
                tensor = load_tensor(get_img_bytes(row), image_size)
                tokens = extractor.patch_tokens(tensor)
                batch_tokens.append(tokens)
                batch_labels.append(str(row[label_col]))
                batch_indices.append(img_idx + bi)
            except Exception:
                pass

        if batch_tokens:
            n_patches    = batch_tokens[0].shape[0]
            all_tokens   = torch.cat(batch_tokens, dim=0).to(device)
            # Use BatchTopK (use_threshold=False) to match training sparsity:
            # selects exactly k * total_patches activations globally across the batch
            all_features = ae.encode(all_tokens, use_threshold=False).cpu()

            for bi, (label, real_idx) in enumerate(zip(batch_labels, batch_indices)):
                start    = bi * n_patches
                features = all_features[start:start + n_patches]  # [n_patches, dict_size]

                # Count patches where each feature fires (non-zero after BatchTopK)
                patch_count = (features > 0).float().sum(0).numpy()   # [dict_size]
                img_act     = features.max(0).values.numpy()           # peak activation value

                # Feature "active" in this image if fires on >= threshold% of patches
                min_patches = max(1, int(n_patches * threshold))
                active = patch_count >= min_patches
                activation_count += active.astype(np.float64)
                activation_sum   += np.where(active, img_act, 0.0)
                label_act_sum[label]   += img_act
                label_act_count[label] += active.astype(np.float64)

                for li in np.where(active)[0]:
                    val = float(img_act[li])
                    top_refs_by_class[li][label].append((val, real_idx))
                    top_refs_global[li].append((val, real_idx))

        if img_idx % 200 == 0:
            print(f"  {img_idx}/{n_images}")
        img_idx += encode_batch_size

    activated_frequency = activation_count / n_images
    mean_activation = np.where(
        activation_count > 0,
        activation_sum / activation_count,
        0.0,
    )

    all_labels = list(label_act_sum.keys())
    label_matrix = np.stack([label_act_sum[l] for l in all_labels], axis=0)
    col_sums = label_matrix.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    label_probs = label_matrix / col_sums
    eps = 1e-10
    label_entropy = -(label_probs * np.log(label_probs + eps)).sum(axis=0)
    label_entropy = np.where(activation_count > 0, label_entropy, 0.0)

    top_refs_by_class_sorted = []
    for li in range(dict_size):
        d = {}
        for lbl, refs in top_refs_by_class[li].items():
            d[lbl] = sorted(refs, reverse=True)[:top_k_refs]
        top_refs_by_class_sorted.append(d)

    top_refs_global_sorted = [
        sorted(refs, reverse=True)[:top_k_refs * 10]
        for refs in top_refs_global
    ]

    strongest_class = []
    for li in range(dict_size):
        best_label, best_val = None, -1.0
        for lbl, refs in top_refs_by_class[li].items():
            total = sum(v for v, _ in refs)
            if total > best_val:
                best_val = total
                best_label = lbl
        strongest_class.append(best_label)

    return {
        "activated_frequency": activated_frequency,
        "mean_activation":     mean_activation,
        "label_entropy":       label_entropy,
        "activation_count":    activation_count,
        "top_refs_by_class":   top_refs_by_class_sorted,
        "top_refs_global":     top_refs_global_sorted,
        "strongest_class":     strongest_class,
        "all_labels":          all_labels,
    }


# ── Step 2: Select latents ────────────────────────────────────

def select_latents(stats, n_latents, min_count=5, dict_size=16384, hl_fraction=0.25):
    """
    Select top n_latents from G0 (class-specific) and G3 (shared) for visualization.
    """
    mean_a = stats["mean_activation"]
    count  = stats["activation_count"]
    group_size = int(dict_size * hl_fraction)

    # G0: features 0 ~ group_size-1
    hl_valid  = count[:group_size] >= min_count
    hl_idx    = np.where(hl_valid)[0]
    hl_top    = hl_idx[np.argsort(mean_a[:group_size][hl_valid])[-n_latents:]][::-1]

    # G3: features 3*group_size ~ dict_size-1
    start_g3  = 3 * group_size
    ll_valid  = count[start_g3:] >= min_count
    ll_idx    = np.where(ll_valid)[0] + start_g3
    ll_top    = ll_idx[np.argsort(mean_a[start_g3:][ll_valid])[-n_latents:]][::-1]

    print(f"[Select] G0 active latents: {hl_valid.sum()}, showing top {n_latents}")
    print(f"[Select] G3 active latents: {ll_valid.sum()}, showing top {n_latents}")

    return {
        "class_specific": hl_top.tolist(),
        "shared":         ll_top.tolist(),
    }


# ── Step 3: Big grid figure ───────────────────────────────────

@torch.no_grad()
def plot_big_grid(
    latent_indices, mode, stats, df, extractor, ae,
    device, image_size, outdir, n_images_per_latent=10,
):
    n_latents = len(latent_indices)
    n_cols = n_images_per_latent * 2

    fig, axes = plt.subplots(
        n_latents, n_cols,
        figsize=(2.2 * n_cols, 3.0 * n_latents),
        squeeze=False,
    )

    for row_idx, latent_idx in enumerate(latent_indices):
        freq    = stats["activated_frequency"][latent_idx]
        mean_v  = stats["mean_activation"][latent_idx]
        entropy = stats["label_entropy"][latent_idx]

        if mode == "class_specific":
            strongest  = stats["strongest_class"][latent_idx]
            class_refs = stats["top_refs_by_class"][latent_idx]
            refs = class_refs.get(strongest, stats["top_refs_global"][latent_idx])[:n_images_per_latent]
            show_label = f"class={strongest}"
        else:
            class_refs = stats["top_refs_by_class"][latent_idx]
            all_labels = list(class_refs.keys())
            np.random.shuffle(all_labels)
            refs = []
            for lbl in all_labels:
                if len(refs) >= n_images_per_latent:
                    break
                if class_refs[lbl]:
                    refs.append(class_refs[lbl][0])
            for item in stats["top_refs_global"][latent_idx]:
                if len(refs) >= n_images_per_latent:
                    break
                if item not in refs:
                    refs.append(item)
            show_label = f"{len(all_labels[:n_images_per_latent])} classes"

        col_idx = 0
        for ref_item in refs:
            if col_idx >= n_cols:
                break
            val, img_idx = ref_item
            try:
                row = df.iloc[img_idx]
                img_bytes  = get_img_bytes(row)
                pil        = load_pil(img_bytes, image_size)
                tensor     = load_tensor(img_bytes, image_size)
                tokens     = extractor.patch_tokens(tensor)
                features   = ae.encode(tokens.to(device), use_threshold=False).cpu()
                side       = int(math.sqrt(tokens.shape[0]))
                fmap       = features[:, latent_idx].view(side, side).numpy()
                fmap_norm  = minmax_norm(fmap)
                label_name = str(row["label"])

                axes[row_idx, col_idx].imshow(pil)
                axes[row_idx, col_idx].set_title(label_name, fontsize=5, pad=1)
                axes[row_idx, col_idx].axis("off")

                axes[row_idx, col_idx + 1].imshow(pil)
                axes[row_idx, col_idx + 1].imshow(
                    fmap_norm, alpha=0.6, interpolation="bilinear",
                    extent=(0, image_size, image_size, 0),
                    cmap="Reds", vmin=0, vmax=1,
                )
                axes[row_idx, col_idx + 1].axis("off")
                col_idx += 2
            except Exception:
                axes[row_idx, col_idx].axis("off")
                if col_idx + 1 < n_cols:
                    axes[row_idx, col_idx + 1].axis("off")
                col_idx += 2

        while col_idx < n_cols:
            axes[row_idx, col_idx].axis("off")
            col_idx += 1

        axes[row_idx, 0].set_ylabel(
            f"C{latent_idx}\n{show_label}\n"
            f"f={freq:.3f} μ={mean_v:.3f}\nH={entropy:.2f}",
            fontsize=6, rotation=0, labelpad=70, va="center",
        )

    mode_title = {
        "class_specific": "G0 Concepts (dist=4, large-scale contrastive)\n"
                          "Each row = one concept | All images from strongest class",
        "shared":         "G3 Concepts (recon-only, no contrastive)\n"
                          "Each row = one concept | Images from many classes",
    }
    plt.suptitle(mode_title[mode], fontsize=12, y=1.01)
    plt.tight_layout()
    path = outdir / f"{mode}_concepts_grid.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Scatter plot ──────────────────────────────────────────────

def plot_scatter(stats, outdir, min_count=5, dict_size=16384, hl_fraction=0.25):
    """
    Two scatter plots:
      Left:  label entropy (red=class-specific, blue=shared)
      Right: group membership (G0–G3 in different colors)
    """
    freq    = stats["activated_frequency"]
    mean_a  = stats["mean_activation"]
    entropy = stats["label_entropy"]
    count   = stats["activation_count"]

    valid   = count >= min_count
    indices = np.where(valid)[0]

    log_freq = np.log10(freq[indices] + 1e-10)
    log_mean = np.log10(mean_a[indices] + 1e-10)
    ent_v    = entropy[indices]

    group_size    = int(dict_size * hl_fraction)
    group_colors  = ["#F97316", "#7C3AED", "#16A34A", "#DB2777"]
    group_labels  = [
        f"G0 dist=4 (0~{group_size-1})",
        f"G1 dist=2 ({group_size}~{2*group_size-1})",
        f"G2 dist=1 ({2*group_size}~{3*group_size-1})",
        f"G3 recon-only ({3*group_size}~{dict_size-1})",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: label entropy
    ax = axes[0]
    sc = ax.scatter(log_freq, log_mean, c=ent_v, cmap="coolwarm_r",
                    alpha=0.5, s=3, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Label Entropy")
    ax.set_xlabel("Log₁₀ Activated Frequency", fontsize=11)
    ax.set_ylabel("Log₁₀ Mean Activation Value", fontsize=11)
    ax.set_title("Label Entropy\nRed = class-specific | Blue = shared", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: group membership
    ax = axes[1]
    for g in range(4):
        start = g * group_size
        end   = (g + 1) * group_size if g < 3 else dict_size
        mask  = (indices >= start) & (indices < end)
        ax.scatter(
            log_freq[mask], log_mean[mask],
            color=group_colors[g], alpha=0.5, s=4, linewidths=0,
            label=f"{group_labels[g]} (n={mask.sum()})",
        )
    ax.set_xlabel("Log₁₀ Activated Frequency", fontsize=11)
    ax.set_ylabel("Log₁₀ Mean Activation Value", fontsize=11)
    ax.set_title("Multi-scale SAE Groups\nG0=large-scale (dist=4) → G3=recon-only", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("SAE Latent Statistics Scatter Plot", fontsize=13, y=1.01)
    plt.tight_layout()
    path = outdir / "latent_statistics_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",             type=str, required=True)
    parser.add_argument("--ckpt",                type=str, required=True)
    parser.add_argument("--n_images",            type=int, default=2000)
    parser.add_argument("--threshold",           type=float, default=0.02,
                        help="Min fraction of patches that must fire for a feature to be 'active' in an image (default 0.02 = ~5 patches)")
    parser.add_argument("--n_latents",           type=int, default=5)
    parser.add_argument("--n_images_per_latent", type=int, default=10)
    parser.add_argument("--top_k_refs",          type=int, default=50)
    parser.add_argument("--encode_batch_size",   type=int, default=8)
    parser.add_argument("--dino_model",  type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--dict_size",   type=int, default=16384)
    parser.add_argument("--k",           type=int, default=32)
    parser.add_argument("--hl_fraction", type=float, default=0.25)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--outdir",      type=str, default="./latent_statistics_v3")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    print(f"[Data] Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    if len(df) > args.n_images:
        df = df.sample(args.n_images, random_state=42).reset_index(drop=True)
    print(f"[Data] {len(df)} images, {df['label'].nunique()} classes")

    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)
    sample_tensor = load_tensor(get_img_bytes(df.iloc[0]), args.image_size)
    with torch.no_grad():
        activation_dim = extractor.patch_tokens(sample_tensor).shape[-1]

    print(f"[Model] Loading SAE from {args.ckpt}...")
    ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    stats = collect_latent_statistics(
        df=df, label_col="label",
        extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, dict_size=args.dict_size,
        threshold=args.threshold, top_k_refs=args.top_k_refs,
        encode_batch_size=args.encode_batch_size,
    )

    active = (stats["activation_count"] >= 5).sum()
    print(f"\n[Summary] Active latents: {active} / {args.dict_size}")

    plot_scatter(stats, outdir, dict_size=args.dict_size, hl_fraction=args.hl_fraction)

    selected = select_latents(stats, n_latents=args.n_latents,
                               dict_size=args.dict_size, hl_fraction=args.hl_fraction)

    print(f"\n[Select] G0 (class_specific) latents: {selected['class_specific']}")
    for li in selected['class_specific']:
        print(f"  C{li}: freq={stats['activated_frequency'][li]:.4f}  "
              f"mean={stats['mean_activation'][li]:.4f}  "
              f"entropy={stats['label_entropy'][li]:.3f}  "
              f"strongest_class={stats['strongest_class'][li]}")

    print(f"\n[Select] G3 (shared) latents: {selected['shared']}")
    for li in selected['shared']:
        print(f"  C{li}: freq={stats['activated_frequency'][li]:.4f}  "
              f"mean={stats['mean_activation'][li]:.4f}  "
              f"entropy={stats['label_entropy'][li]:.3f}")

    print(f"\n[Plot] G0 class-specific grid...")
    plot_big_grid(
        latent_indices=selected["class_specific"], mode="class_specific",
        stats=stats, df=df, extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, outdir=outdir,
        n_images_per_latent=args.n_images_per_latent,
    )

    print(f"\n[Plot] G3 shared concept grid...")
    plot_big_grid(
        latent_indices=selected["shared"], mode="shared",
        stats=stats, df=df, extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, outdir=outdir,
        n_images_per_latent=args.n_images_per_latent,
    )

    print(f"\n[Done] Saved to: {outdir}")
    print("  latent_statistics_scatter.png")
    print("  class_specific_concepts_grid.png")
    print("  shared_concepts_grid.png")


if __name__ == "__main__":
    main()
