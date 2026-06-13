"""
analyze_discriminative_concepts.py

Finds the most DISCRIMINATIVE concepts for each class using linear probes,
following the methodology of Fel et al. (Rabbit Hull, 2026).

Pipeline:
  1. Encode all images through DINOv2 + SAE → concept activation vectors [dict_size]
  2. Train a multinomial logistic regression (linear probe) on these vectors
  3. Extract per-class weights → top-K concepts with highest weight = most discriminative
  4. Visualize these concepts on example images from that class
  5. Plot a concept-class discriminability heatmap

This is more meaningful than activation strength because:
  - Activation strength: "which concepts fire a lot on this class"
  - Discriminative weight: "which concepts distinguish this class from all others"

Usage:
    python analyze_discriminative_concepts.py \
        --parquet ./imagenet_data/train-00000-of-00001-*.parquet \
        --ckpt ./checkpoints_imagenet/ae_final.pt \
        --n_classes 20 \
        --top_k 10 \
        --n_images_per_class 100 \
        --device cuda \
        --outdir ./discriminative_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from io import BytesIO
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE


# ── DINOv2 ───────────────────────────────────────────────────

class DINOFeatureExtractor:
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = device
        os.environ["TORCH_HOME"] = "/home/ubuntu/.cache/torch"
        self.model = torch.hub.load(
            "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
            model_name,
            source="local",
            trust_repo=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """[1, 3, H, W] → [N, D]"""
        feats = self.model.forward_features(image_tensor.to(self.device))
        return feats["x_norm_patchtokens"].squeeze(0)  # [N, D]


# ── SAE ──────────────────────────────────────────────────────

def load_sae(ckpt: str, activation_dim: int, dict_size: int, k: int, device: str):
    group_fractions = [0.25, 0.25, 0.25, 0.25]
    group_sizes = [int(f * dict_size) for f in group_fractions[:-1]]
    group_sizes.append(dict_size - sum(group_sizes))
    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=dict_size,
        k=k,
        group_sizes=group_sizes,
    ).to(device)
    ae.load_state_dict(torch.load(ckpt, map_location=device))
    ae.eval()
    return ae


# ── Image utilities ───────────────────────────────────────────

def load_image_tensor(img_bytes: bytes, image_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)


def load_pil(img_bytes: bytes, image_size: int) -> Image.Image:
    return Image.open(BytesIO(img_bytes)).convert("RGB").resize((image_size, image_size))


def minmax_norm(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


# ── Step 1: Encode all images → concept vectors ───────────────

@torch.no_grad()
def encode_dataset(
    df,
    label_col: str,
    selected_labels: list,
    extractor: DINOFeatureExtractor,
    ae,
    device: str,
    image_size: int,
    n_images_per_class: int,
    dict_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Encode images through DINOv2 + SAE.
    For each image, pool patch-level features → image-level concept vector.

    Returns:
        X: [N_images, dict_size]  - concept activation vectors
        y: [N_images]             - integer class labels
        label_to_idx: dict        - maps label string → integer
    """
    label_to_idx = {str(l): i for i, l in enumerate(selected_labels)}

    X_list, y_list = [], []

    for label in selected_labels:
        rows = df[df[label_col] == label]
        rows = rows.sample(min(n_images_per_class, len(rows)), random_state=42)

        for _, row in rows.iterrows():
            try:
                img_data = row["image"]
                img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data

                tensor = load_image_tensor(img_bytes, image_size)
                tokens = extractor.patch_tokens(tensor)          # [N, D]
                features = ae.encode(tokens.to(device)).cpu()    # [N, F]

                # Mean pooling over patches → image-level concept vector
                img_vec = features.mean(dim=0).numpy()           # [F]

                X_list.append(img_vec)
                y_list.append(label_to_idx[str(label)])

            except Exception:
                continue

        print(f"  [{label}] encoded {len([y for y in y_list if y == label_to_idx[str(label)]])} images")

    X = np.stack(X_list, axis=0)  # [N, F]
    y = np.array(y_list)          # [N]
    return X, y, label_to_idx


# ── Step 2: Linear probe ──────────────────────────────────────

def train_linear_probe(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    
    clf = LogisticRegression(
        max_iter=1000,
        C=0.1,
        solver="saga",
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)
    
    train_acc = clf.score(X_train_scaled, y_train)
    val_acc   = clf.score(X_val_scaled, y_val)
    
    print(f"[Probe] Train accuracy: {train_acc:.3f}")
    print(f"[Probe] Val accuracy:   {val_acc:.3f}")
    print("\n[Probe] Per-class report (val):")
    print(classification_report(y_val, clf.predict(X_val_scaled)))
    
    return clf, scaler, val_acc


# ── Step 3: Extract discriminative concepts per class ─────────

def get_discriminative_concepts(
    clf: LogisticRegression,
    top_k: int,
    selected_labels: list,
    label_to_idx: dict,
) -> dict[str, dict]:
    """
    Extract top-K most discriminative concepts per class from
    the linear probe weight matrix.

    clf.coef_ has shape [n_classes, dict_size]:
      - positive weight → concept helps identify this class
      - negative weight → concept helps rule out this class

    We take top-K by absolute weight to capture both directions,
    but also report sign so we know if it's a positive or negative indicator.
    """
    results = {}
    coef = clf.coef_  # [n_classes, dict_size]

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    for class_idx in range(len(selected_labels)):
        weights = coef[class_idx]  # [dict_size]

        # Top-K by absolute value
        abs_weights = np.abs(weights)
        top_indices = np.argsort(abs_weights)[::-1][:top_k]
        top_weights = weights[top_indices]

        label = idx_to_label[class_idx]
        results[label] = {
            "top_concept_indices": top_indices.tolist(),
            "top_concept_weights": top_weights.tolist(),
            # Positive = concept is a positive indicator for this class
            "is_positive_indicator": (top_weights > 0).tolist(),
        }

    return results


# ── Step 4: Visualization ─────────────────────────────────────

def plot_discriminability_heatmap(
    clf: LogisticRegression,
    selected_labels: list,
    label_to_idx: dict,
    discriminative_results: dict,
    top_k: int,
    outdir: Path,
):
    """
    Heatmap: rows = classes, cols = union of top-K discriminative concepts.
    Color = linear probe weight (signed: red = positive, blue = negative).

    This is the key figure: shows which concepts are class-specific
    vs shared, and whether they are positive or negative indicators.
    """
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Collect union of top-k concepts
    all_concepts = set()
    for info in discriminative_results.values():
        all_concepts.update(info["top_concept_indices"])
    concept_list = sorted(all_concepts)

    # Build signed weight matrix
    n_classes = len(selected_labels)
    matrix = np.zeros((n_classes, len(concept_list)))
    coef = clf.coef_

    for class_idx in range(n_classes):
        for j, cidx in enumerate(concept_list):
            matrix[class_idx, j] = coef[class_idx, cidx]

    # Plot
    fig, ax = plt.subplots(
        figsize=(max(14, len(concept_list) * 0.45), max(6, n_classes * 0.45))
    )
    vmax = np.abs(matrix).max()
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Linear probe weight\n(red=positive indicator, blue=negative)")

    labels_ordered = [idx_to_label[i] for i in range(n_classes)]
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(labels_ordered, fontsize=8)
    ax.set_xticks(range(len(concept_list)))
    ax.set_xticklabels([f"C{c}" for c in concept_list], fontsize=6, rotation=90)
    ax.set_xlabel("Concept index")
    ax.set_ylabel("Class")
    ax.set_title(
        f"Concept discriminability per class (linear probe weights)\n"
        f"Red = concept helps identify this class | Blue = concept rules out this class"
    )

    plt.tight_layout()
    path = outdir / "discriminability_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_top_concepts_per_class(
    df,
    label_col: str,
    label: str,
    concept_indices: list[int],
    concept_weights: list[float],
    is_positive: list[bool],
    extractor: DINOFeatureExtractor,
    ae,
    device: str,
    image_size: int,
    outdir: Path,
    n_example_images: int = 4,
    top_k_show: int = 6,
):
    """
    For one class, show the top discriminative concepts overlaid on images.
    Only show top_k_show concepts (fewer = clearer figure).
    Title each concept with its weight and sign.
    """
    concept_indices = concept_indices[:top_k_show]
    concept_weights = concept_weights[:top_k_show]
    is_positive = is_positive[:top_k_show]

    rows = df[df[label_col].astype(str) == str(label)].sample(
        min(n_example_images, max(1, len(df[df[label_col].astype(str) == str(label)]))),
        random_state=1
    )

    if len(rows) == 0:
        print(f"  Warning: no images found for class {label}, skipping")
        return

    n_imgs = len(rows)
    n_concepts = len(concept_indices)

    fig, axes = plt.subplots(
        n_imgs, n_concepts + 1,
        figsize=(2.8 * (n_concepts + 1), 2.8 * n_imgs)
    )
    if n_imgs == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (_, row) in enumerate(rows.iterrows()):
        try:
            img_data = row["image"]
            img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data

            pil = load_pil(img_bytes, image_size)
            tensor = load_image_tensor(img_bytes, image_size)

            with torch.no_grad():
                tokens = extractor.patch_tokens(tensor)
                features = ae.encode(tokens.to(device)).cpu()

            side = int(math.sqrt(tokens.shape[0]))

            # Original
            axes[row_idx, 0].imshow(pil)
            axes[row_idx, 0].set_title("Original" if row_idx == 0 else "", fontsize=8)
            axes[row_idx, 0].axis("off")

            # Each concept
            for col_idx, (cidx, weight, pos) in enumerate(
                zip(concept_indices, concept_weights, is_positive)
            ):
                fmap = features[:, cidx].view(side, side)
                fmap_norm = minmax_norm(fmap).numpy()

                cmap = "Reds" if pos else "Blues"
                axes[row_idx, col_idx + 1].imshow(pil)
                axes[row_idx, col_idx + 1].imshow(
                    fmap_norm, alpha=0.6, interpolation="bilinear",
                    extent=(0, image_size, image_size, 0),
                    cmap=cmap, vmin=0, vmax=1,
                )
                axes[row_idx, col_idx + 1].axis("off")

                if row_idx == 0:
                    sign = "+" if pos else "-"
                    axes[row_idx, col_idx + 1].set_title(
                        f"C{cidx}\n({sign}{abs(weight):.2f})", fontsize=7
                    )

        except Exception as e:
            print(f" Warning:{e}")
            continue

    safe_label = str(label).replace("/", "_").replace(" ", "_")
    plt.suptitle(
        f"Class: {label} — Top discriminative concepts\n"
        f"Red overlay = positive indicator | Blue overlay = negative indicator",
        fontsize=10
    )
    plt.tight_layout()
    path = outdir / f"class_{safe_label}_discriminative.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",  type=str, required=True)
    parser.add_argument("--ckpt",     type=str, required=True)
    parser.add_argument("--n_classes",          type=int, default=20)
    parser.add_argument("--top_k",              type=int, default=10)
    parser.add_argument("--n_images_per_class", type=int, default=100)
    parser.add_argument("--n_example_images",   type=int, default=4)
    parser.add_argument("--top_k_show",         type=int, default=6,
                        help="Concepts to show in overlay plots (<=top_k)")
    parser.add_argument("--dino_model",  type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--dict_size",   type=int, default=16384)
    parser.add_argument("--k",           type=int, default=64)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--outdir",      type=str, default="./discriminative_analysis")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────
    import pandas as pd
    print(f"[Data] Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    label_col = "label"

    all_labels = df[label_col].unique().tolist()
    selected_labels = random.sample(all_labels, min(args.n_classes, len(all_labels)))
    print(f"[Data] {len(selected_labels)} classes selected")

    # ── 2. Load models ───────────────────────────────────────
    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    sample_row = df.iloc[0]
    img_data = sample_row["image"]
    img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
    sample_tensor = load_image_tensor(img_bytes, args.image_size)
    with torch.no_grad():
        sample_tokens = extractor.patch_tokens(sample_tensor)
    activation_dim = sample_tokens.shape[-1]

    print(f"[Model] Loading SAE from {args.ckpt}...")
    ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    # ── 3. Encode dataset ────────────────────────────────────
    print(f"\n[Encode] Encoding images ({args.n_images_per_class}/class)...")
    X, y, label_to_idx = encode_dataset(
        df=df,
        label_col=label_col,
        selected_labels=selected_labels,
        extractor=extractor,
        ae=ae,
        device=args.device,
        image_size=args.image_size,
        n_images_per_class=args.n_images_per_class,
        dict_size=args.dict_size,
    )
    print(f"[Encode] Dataset shape: X={X.shape}, y={y.shape}")

    # ── 4. Linear probe ──────────────────────────────────────
    clf, scaler, acc = train_linear_probe(X, y)

    # ── 5. Discriminative concepts ───────────────────────────
    discriminative_results = get_discriminative_concepts(
        clf, args.top_k, selected_labels, label_to_idx
    )

    # Save JSON
    json_path = outdir / "discriminative_concepts.json"
    with open(json_path, "w") as f:
        json.dump({
            "probe_accuracy": acc,
            "n_classes": len(selected_labels),
            "n_images_per_class": args.n_images_per_class,
            "results": discriminative_results,
        }, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Print summary
    print(f"\n[Results] Linear probe accuracy: {acc:.3f}")
    print(f"[Results] Top-{args.top_k} discriminative concepts per class:")
    for label, info in discriminative_results.items():
        signs = ["+" if p else "-" for p in info["is_positive_indicator"]]
        pairs = [f"C{c}({s}{abs(w):.2f})"
                 for c, w, s in zip(
                     info["top_concept_indices"],
                     info["top_concept_weights"],
                     signs
                 )]
        print(f"  {label}: {pairs}")

    # ── 6. Heatmap ───────────────────────────────────────────
    print("\n[Plot] Discriminability heatmap...")
    plot_discriminability_heatmap(
        clf, selected_labels, label_to_idx,
        discriminative_results, args.top_k, outdir
    )

    # ── 7. Per-class overlay plots ───────────────────────────
    print("\n[Plot] Per-class discriminative concept overlays...")
    for label, info in discriminative_results.items():
        print(f"  Class: {label}")
        plot_top_concepts_per_class(
            df=df,
            label_col=label_col,
            label=label,
            concept_indices=info["top_concept_indices"],
            concept_weights=info["top_concept_weights"],
            is_positive=info["is_positive_indicator"],
            extractor=extractor,
            ae=ae,
            device=args.device,
            image_size=args.image_size,
            outdir=outdir,
            n_example_images=args.n_example_images,
            top_k_show=args.top_k_show,
        )

    print(f"\n[Done] All results in: {outdir}")


if __name__ == "__main__":
    main()