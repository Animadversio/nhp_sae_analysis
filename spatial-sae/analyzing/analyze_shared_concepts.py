"""
analyze_shared_concepts.py

Finds concepts that are strongly activated across MULTIPLE classes,
and visualizes where each shared concept activates on example images
from each of those classes.

Input:
  - discriminative_concepts.json  (from analyze_discriminative_concepts.py)
  - parquet file
  - SAE checkpoint

Output (per shared concept):
  concept_{idx}_shared_across_classes.png
    Layout: rows = classes, cols = example images
    Shows where concept activates on real images from each class.

Usage:
    python analyze_shared_concepts.py \
        --json ./discriminative_analysis/discriminative_concepts.json \
        --parquet ./imagenet_data/train-00000-of-00001-*.parquet \
        --ckpt ./checkpoints_imagenet/ae_final.pt \
        --min_classes 3 \
        --top_n_concepts 10 \
        --n_images_per_class 3 \
        --device cuda \
        --outdir ./shared_concept_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE


# ── Models ────────────────────────────────────────────────────

class DINOFeatureExtractor:
    def __init__(self, model_name="dinov2_vitb14", device="cuda"):
        self.device = device
        os.environ["TORCH_HOME"] = "/home/ubuntu/.cache/torch"
        self.model = torch.hub.load(
            "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
            model_name, source="local", trust_repo=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, tensor):
        feats = self.model.forward_features(tensor.to(self.device))
        return feats["x_norm_patchtokens"].squeeze(0)  # [N, D]


def load_sae(ckpt, activation_dim, dict_size, k, device):
    fracs = [0.25, 0.25, 0.25, 0.25]
    sizes = [int(f * dict_size) for f in fracs[:-1]]
    sizes.append(dict_size - sum(sizes))
    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim, dict_size=dict_size,
        k=k, group_sizes=sizes,
    ).to(device)
    ae.load_state_dict(torch.load(ckpt, map_location=device))
    ae.eval()
    return ae


# ── Image utils ───────────────────────────────────────────────

def get_img_bytes(row):
    img_data = row["image"]
    return img_data["bytes"] if isinstance(img_data, dict) else img_data

def load_tensor(img_bytes, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return transform(Image.open(BytesIO(img_bytes)).convert("RGB")).unsqueeze(0)

def load_pil(img_bytes, image_size):
    return Image.open(BytesIO(img_bytes)).convert("RGB").resize((image_size, image_size))

def minmax_norm(x):
    lo, hi = x.min(), x.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


# ── Step 1: Find shared concepts ──────────────────────────────

def find_shared_concepts(
    json_data: dict,
    min_classes: int,
) -> list[tuple[int, list[str], list[float]]]:
    """
    Find concepts that appear in the top-K list of at least min_classes classes.

    Returns list of (concept_idx, [class_labels], [abs_weights]) sorted by
    number of classes (descending), then total weight (descending).
    """
    # concept_idx → {class_label: abs_weight}
    concept_to_classes: dict[int, dict[str, float]] = defaultdict(dict)

    results = json_data["results"]
    for label, info in results.items():
        for cidx, weight in zip(
            info["top_concept_indices"],
            info["top_concept_weights"],
        ):
            concept_to_classes[cidx][label] = abs(weight)

    # Filter: only keep concepts in >= min_classes classes
    shared = []
    for cidx, class_weights in concept_to_classes.items():
        if len(class_weights) >= min_classes:
            labels = list(class_weights.keys())
            weights = [class_weights[l] for l in labels]
            shared.append((cidx, labels, weights))

    # Sort: most classes first, then highest total weight
    shared.sort(key=lambda x: (len(x[1]), sum(x[2])), reverse=True)
    return shared


# ── Step 2: Visualize each shared concept ─────────────────────

@torch.no_grad()
def visualize_shared_concept(
    concept_idx: int,
    class_labels: list[str],
    class_weights: list[float],
    df,
    label_col: str,
    extractor: DINOFeatureExtractor,
    ae,
    device: str,
    image_size: int,
    n_images_per_class: int,
    outdir: Path,
):
    """
    For one concept, show its activation heatmap on example images
    from each class where it appears.

    Layout:
      rows = classes
      cols = example images from that class
    Each cell shows the image with the concept heatmap overlaid.
    """
    n_classes = len(class_labels)
    n_cols = n_images_per_class

    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(3 * n_cols, 3 * n_classes),
        squeeze=False,
    )

    for row_idx, (label, weight) in enumerate(zip(class_labels, class_weights)):
        # Sample images for this class
        mask = df[label_col].astype(str) == str(label)
        rows = df[mask]
        if len(rows) == 0:
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].axis("off")
            continue

        rows = rows.sample(min(n_cols, len(rows)), random_state=row_idx)

        col_idx = 0
        for _, row in rows.iterrows():
            if col_idx >= n_cols:
                break
            try:
                img_bytes = get_img_bytes(row)
                pil = load_pil(img_bytes, image_size)
                tensor = load_tensor(img_bytes, image_size)

                tokens = extractor.patch_tokens(tensor)        # [N, D]
                features = ae.encode(tokens.to(device)).cpu()  # [N, F]
                side = int(math.sqrt(tokens.shape[0]))

                fmap = features[:, concept_idx].view(side, side)
                fmap_norm = minmax_norm(fmap).numpy()

                ax = axes[row_idx, col_idx]
                ax.imshow(pil)
                ax.imshow(
                    fmap_norm, alpha=0.55, interpolation="bilinear",
                    extent=(0, image_size, image_size, 0),
                    cmap="Reds", vmin=0, vmax=1,
                )
                ax.axis("off")

                # Label on first column only
                if col_idx == 0:
                    ax.set_ylabel(
                        f"class {label}\n(w={weight:.3f})",
                        fontsize=8, rotation=0, labelpad=60, va="center",
                    )

                col_idx += 1

            except Exception as e:
                axes[row_idx, col_idx].axis("off")
                col_idx += 1
                continue

        # Fill empty cells
        while col_idx < n_cols:
            axes[row_idx, col_idx].axis("off")
            col_idx += 1

    plt.suptitle(
        f"Concept {concept_idx} — shared across {n_classes} classes\n"
        f"Red = where this concept activates on each image",
        fontsize=11,
    )
    plt.tight_layout()

    path = outdir / f"concept_{concept_idx}_shared_{n_classes}classes.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",    type=str, required=True,
                        help="Path to discriminative_concepts.json")
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--ckpt",    type=str, required=True)
    parser.add_argument("--min_classes",       type=int, default=3,
                        help="Minimum number of classes a concept must appear in")
    parser.add_argument("--top_n_concepts",    type=int, default=10,
                        help="How many shared concepts to visualize")
    parser.add_argument("--n_images_per_class", type=int, default=3,
                        help="Example images per class per concept")
    parser.add_argument("--dino_model",  type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--dict_size",   type=int, default=16384)
    parser.add_argument("--k",           type=int, default=64)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--outdir",      type=str, default="./shared_concept_analysis")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load JSON ─────────────────────────────────────────
    with open(args.json) as f:
        json_data = json.load(f)
    print(f"[Data] Loaded {len(json_data['results'])} classes from JSON")

    # ── 2. Find shared concepts ───────────────────────────────
    shared = find_shared_concepts(json_data, args.min_classes)
    print(f"\n[Shared] Found {len(shared)} concepts appearing in "
          f">= {args.min_classes} classes")

    if not shared:
        print("No shared concepts found. Try lowering --min_classes.")
        return

    # Print summary table
    print(f"\n{'Concept':>8} | {'# Classes':>9} | {'Total Weight':>12} | Classes")
    print("-" * 70)
    for cidx, labels, weights in shared[:args.top_n_concepts]:
        print(f"{cidx:>8} | {len(labels):>9} | {sum(weights):>12.4f} | {labels}")

    # Save summary JSON
    summary = [
        {
            "concept_idx": cidx,
            "n_classes": len(labels),
            "total_weight": sum(weights),
            "classes": labels,
            "weights": weights,
        }
        for cidx, labels, weights in shared
    ]
    with open(outdir / "shared_concepts_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── 3. Load models ───────────────────────────────────────
    import pandas as pd
    print(f"\n[Data] Loading parquet: {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    label_col = "label"

    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    sample_bytes = get_img_bytes(df.iloc[0])
    sample_tensor = load_tensor(sample_bytes, args.image_size)
    with torch.no_grad():
        sample_tokens = extractor.patch_tokens(sample_tensor)
    activation_dim = sample_tokens.shape[-1]

    print(f"[Model] Loading SAE from {args.ckpt}...")
    ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    # ── 4. Visualize top shared concepts ─────────────────────
    print(f"\n[Plot] Visualizing top {args.top_n_concepts} shared concepts...")
    for cidx, labels, weights in shared[:args.top_n_concepts]:
        print(f"\n  Concept {cidx} — {len(labels)} classes: {labels}")
        visualize_shared_concept(
            concept_idx=cidx,
            class_labels=labels,
            class_weights=weights,
            df=df,
            label_col=label_col,
            extractor=extractor,
            ae=ae,
            device=args.device,
            image_size=args.image_size,
            n_images_per_class=args.n_images_per_class,
            outdir=outdir,
        )

    print(f"\n[Done] Results saved to: {outdir}")
    print(f"  shared_concepts_summary.json  — ranked list of shared concepts")
    print(f"  concept_XXX_shared_Yclasses.png — activation maps per concept")


if __name__ == "__main__":
    main()