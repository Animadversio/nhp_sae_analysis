"""
analyze_augmentation_probe.py

Linear probe classification with augmented images.
Each original image is encoded under the original view and 7 transformations
(same as analyze_transformation_invariance.py). All versions carry the same
class label, testing whether SAE features allow class discrimination even
across transformations.

Split is image-level (all transforms of an image go to the same train/val
partition) to prevent within-image leakage.

Statistical inference follows analyze_highlevel_vs_lowlevel.py:
  - n_repeats seeds, same images + same split for all feature subsets per seed
  - Paired t-test, Wilcoxon signed-rank, bootstrap 95% CI vs Raw DINO
  - repeat_results.csv, significance_summary.json,
    probe_accuracy_repeated.png

Usage:
    python -u scripts/analyze_augmentation_probe.py \\
        --parquet imagenet_val_probe.parquet \\
        --ckpt ckpts_multiscale_v1/ae_final.pt \\
        --dict_size 16384 --k 32 \\
        --n_classes 20 --n_images_per_class 100 \\
        --device cuda \\
        --outdir results_augprobe/multiscale_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))

DINO_REPO = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache/hub/facebookresearch_dinov2_main"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision import transforms

from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE


# ── Transforms (same as analyze_transformation_invariance.py) ─

NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])

AUG_TRANSFORMS = {
    "original":    BASE_TRANSFORM,
    "hflip":       transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "rotation_90":  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "rotation_180": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(180, 180)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "rotation_270": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(270, 270)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "scale_0.5":   transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Pad(56, fill=(255, 255, 255)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "scale_2.0":   transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.CenterCrop(224),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "crop_0.5":    transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
        transforms.ToTensor(), NORMALIZE,
    ]),
}
N_TRANSFORMS = len(AUG_TRANSFORMS)
TRANSFORM_NAMES = list(AUG_TRANSFORMS.keys())


# ── Models ─────────────────────────────────────────────────────

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
        return feats["x_norm_patchtokens"].squeeze(0)   # [P, D]


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
    return ae


def get_img_bytes(row):
    d = row["image"]
    return d["bytes"] if isinstance(d, dict) else d


# ── Encoding ───────────────────────────────────────────────────

@torch.no_grad()
def encode_all_augmentations(df, label_col, selected_labels, extractor, ae,
                              device, image_size, n_images_per_class,
                              random_state=42):
    """
    For each sampled image, encode all AUG_TRANSFORMS versions.

    Returns
    -------
    X_sae : (N_img, N_transforms, dict_size)
    X_raw : (N_img, N_transforms, activation_dim)
    y     : (N_img,)  — image-level class labels
    label_to_idx : dict
    """
    label_to_idx = {str(l): i for i, l in enumerate(selected_labels)}

    X_sae_list, X_raw_list, y_list = [], [], []

    for label in selected_labels:
        pool = df[df[label_col] == label]
        rows = pool.sample(
            min(n_images_per_class, len(pool)), random_state=random_state
        )
        for _, row in rows.iterrows():
            try:
                img_bytes = get_img_bytes(row)
                pil = Image.open(BytesIO(img_bytes)).convert("RGB")

                sae_vecs, raw_vecs = [], []
                for tfm_name, tfm in AUG_TRANSFORMS.items():
                    tensor = tfm(pil).unsqueeze(0)
                    tokens = extractor.patch_tokens(tensor)         # [P, D]
                    raw_vecs.append(tokens.mean(dim=0).cpu().numpy())
                    feats = ae.encode(tokens.to(device), use_threshold=False).cpu()
                    sae_vecs.append(feats.mean(dim=0).numpy())

                X_sae_list.append(np.stack(sae_vecs, axis=0))  # (N_transforms, dict_size)
                X_raw_list.append(np.stack(raw_vecs, axis=0))  # (N_transforms, activation_dim)
                y_list.append(label_to_idx[str(label)])
            except Exception:
                continue

        print(f"  [{label}] {sum(1 for yy in y_list if yy == label_to_idx[str(label)])} images")

    X_sae = np.stack(X_sae_list, axis=0)   # (N_img, N_transforms, dict_size)
    X_raw = np.stack(X_raw_list, axis=0)   # (N_img, N_transforms, activation_dim)
    y     = np.array(y_list)               # (N_img,)
    return X_sae, X_raw, y, label_to_idx


def flatten_with_image_split(X_img, y_img, train_img_idx, test_img_idx):
    """
    Given per-image arrays (N_img, N_transforms, D) and an image-level split,
    return flattened train/test arrays where all transforms of an image are
    kept together in the same partition.
    """
    X_tr = X_img[train_img_idx].reshape(-1, X_img.shape[-1])
    y_tr = np.repeat(y_img[train_img_idx], N_TRANSFORMS)
    X_te = X_img[test_img_idx].reshape(-1, X_img.shape[-1])
    y_te = np.repeat(y_img[test_img_idx], N_TRANSFORMS)
    return X_tr, y_tr, X_te, y_te


# ── Probe training ─────────────────────────────────────────────

def train_probe(X_tr, y_tr, X_te, y_te, subset_name, n_components=250):
    """PCA (fit on train only) → StandardScaler → LogisticRegressionCV."""
    if n_components is not None and X_tr.shape[1] > n_components:
        pca = PCA(n_components=n_components)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=5, max_iter=1000, solver="saga", random_state=42,
    )
    clf.fit(X_tr_s, y_tr)

    train_acc = clf.score(X_tr_s, y_tr)
    val_acc   = clf.score(X_te_s,  y_te)
    print(f"  [{subset_name}] train={train_acc:.3f}  val={val_acc:.3f}  C={clf.C_[0]:.4f}")
    return train_acc, val_acc


# ── Statistics and plotting ────────────────────────────────────

def compute_significance(df_results, reference="Raw DINO", n_bootstrap=5000, rng_seed=0):
    """All-pairs significance tests between every subset."""
    from itertools import combinations
    rng = np.random.default_rng(rng_seed)
    subsets = list(df_results['subset'].unique())
    vals_by_subset = {s: df_results[df_results['subset'] == s]['val_acc'].values
                      for s in subsets}

    def _pairwise(a_name, b_name):
        a = vals_by_subset[a_name]
        b = vals_by_subset[b_name]
        n = min(len(a), len(b))
        diff = a[:n] - b[:n]
        t_stat, t_p = scipy_stats.ttest_rel(a[:n], b[:n])
        try:
            w_stat, w_p = scipy_stats.wilcoxon(diff)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        boot_means = np.array([
            rng.choice(diff, size=len(diff), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        return {
            "mean_diff":     float(diff.mean()),
            "sem_diff":      float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "ci_95_lo":      float(ci_lo),
            "ci_95_hi":      float(ci_hi),
            "t_stat":        float(t_stat),
            "t_p":           float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p":    float(w_p),
            "n_seeds":       int(n),
        }

    summary = {"pairwise_comparisons": {}}
    for a, b in combinations(subsets, 2):
        summary["pairwise_comparisons"][f"{a} vs {b}"] = _pairwise(a, b)

    summary["subset_stats"] = {}
    for s in subsets:
        vals = vals_by_subset[s]
        summary["subset_stats"][s] = {
            "mean":    float(vals.mean()),
            "sem":     float(vals.std(ddof=1) / np.sqrt(len(vals))),
            "min":     float(vals.min()),
            "max":     float(vals.max()),
            "n_seeds": int(len(vals)),
        }
    return summary


def plot_repeated_bars(df_results, outdir, n_classes=20):
    subsets = df_results['subset'].unique().tolist()
    means, sems = [], []
    for s in subsets:
        vals = df_results[df_results['subset'] == s]['val_acc'].values
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))

    x = np.arange(len(subsets))
    width = 0.55
    chance = 1.0 / n_classes

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(subsets)), 5))
    ax.bar(x, means, width, yerr=sems, color="#2563EB", alpha=0.8,
           capsize=4, error_kw=dict(elinewidth=1.5))
    ax.axhline(chance, linestyle="--", color="red", linewidth=1.5,
               label=f"Chance (1/{n_classes} = {chance:.2f})")

    for i, (m, s) in enumerate(zip(means, sems)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Augmentation-Robust Linear Probe (mean ± SEM, {len(df_results['seed'].unique())} seeds)\n"
        f"Transforms: {', '.join(TRANSFORM_NAMES)}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = outdir / "probe_accuracy_repeated.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",            type=str, required=True)
    parser.add_argument("--ckpt",               type=str, required=True)
    parser.add_argument("--n_classes",          type=int, default=20)
    parser.add_argument("--n_images_per_class", type=int, default=200)
    parser.add_argument("--n_repeats",          type=int, default=20)
    parser.add_argument("--dino_model",  type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--dict_size",   type=int, default=16384)
    parser.add_argument("--k",           type=int, default=32)
    parser.add_argument("--hl_fraction", type=float, default=0.25)
    parser.add_argument("--n_pca",       type=int, default=250)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--outdir",      type=str, default="results_augprobe")
    args = parser.parse_args()

    import pandas as pd

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Data] Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    label_col = "label"
    all_labels      = df[label_col].unique().tolist()
    selected_labels = all_labels[:args.n_classes]
    df_selected     = df[df[label_col].isin(selected_labels)]
    print(f"[Data] {len(df_selected)} images, {len(selected_labels)} classes")
    print(f"[Transforms] {TRANSFORM_NAMES}")

    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    sample_pil = Image.open(BytesIO(get_img_bytes(df.iloc[0]))).convert("RGB")
    with torch.no_grad():
        activation_dim = extractor.patch_tokens(
            BASE_TRANSFORM(sample_pil).unsqueeze(0)
        ).shape[-1]

    print(f"[Model] Loading SAE from {args.ckpt}...")
    ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    # Group index ranges from ae.group_sizes
    gs = ae.group_sizes.tolist()
    g_bounds = [0] + list(np.cumsum(gs))
    n_groups = len(gs)
    group_indices_list = [np.arange(g_bounds[g], g_bounds[g + 1]) for g in range(n_groups)]
    group_names = [f"G{g} (features {g_bounds[g]}–{g_bounds[g+1]-1})" for g in range(n_groups)]
    all_indices = np.arange(args.dict_size)

    print(f"[Groups] {n_groups} groups: {[len(g) for g in group_indices_list]}")

    # ── Multi-seed repeated evaluation ──
    print(f"\n{'='*60}")
    print(f"REPEATED AUGMENTATION-ROBUST PROBE: {args.n_repeats} seeds")
    print(f"{'='*60}")

    all_rows = []

    for seed in range(args.n_repeats):
        print(f"\n── Seed {seed}/{args.n_repeats - 1} ──")

        # Encode original + all augmentations for this seed's image sample
        X_sae, X_raw, y_img, label_to_idx = encode_all_augmentations(
            df=df_selected, label_col=label_col,
            selected_labels=selected_labels,
            extractor=extractor, ae=ae,
            device=args.device, image_size=args.image_size,
            n_images_per_class=args.n_images_per_class,
            random_state=seed,
        )
        # X_sae: (N_img, N_transforms, dict_size)
        # X_raw: (N_img, N_transforms, activation_dim)

        # Image-level train/val split (80/20), same for all subsets
        rng = np.random.default_rng(seed)
        n_img = len(y_img)
        perm = rng.permutation(n_img)
        n_train = int(n_img * 0.8)
        train_img_idx = perm[:n_train]
        test_img_idx  = perm[n_train:]

        def run_probe(X_feat, name):
            X_tr, y_tr, X_te, y_te = flatten_with_image_split(
                X_feat, y_img, train_img_idx, test_img_idx)
            return train_probe(X_tr, y_tr, X_te, y_te, name, n_components=args.n_pca)

        # Raw DINOv2 baseline
        tr_raw, vl_raw = run_probe(X_raw, "Raw DINO")
        all_rows.append({"seed": seed, "subset": "Raw DINO",
                         "train_acc": tr_raw, "val_acc": vl_raw})

        # Per-group SAE probes
        for g_idx, (g_indices, g_name) in enumerate(zip(group_indices_list, group_names)):
            tr_g, vl_g = run_probe(X_sae[:, :, g_indices], g_name)
            all_rows.append({"seed": seed, "subset": g_name,
                             "train_acc": tr_g, "val_acc": vl_g})

        # Full SAE
        tr_all, vl_all = run_probe(X_sae, "Full SAE")
        all_rows.append({"seed": seed, "subset": "Full SAE",
                         "train_acc": tr_all, "val_acc": vl_all})

    # ── Save CSV ──
    df_results = pd.DataFrame(all_rows)
    csv_path = outdir / "repeat_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Statistics ──
    sig_summary = compute_significance(df_results, reference="Raw DINO")
    json_path = outdir / "significance_summary.json"
    with open(json_path, "w") as f:
        json.dump(sig_summary, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n── Validation accuracy summary ──")
    for s, st in sig_summary["subset_stats"].items():
        print(f"  {s:35s}  {st['mean']:.3f} ± {st['sem']:.3f}")
    print("\n── All-pairs significance ──")
    for s, cmp in sig_summary["pairwise_comparisons"].items():
        stars = "***" if cmp["t_p"] < 0.001 else (
                "**"  if cmp["t_p"] < 0.01  else (
                "*"   if cmp["t_p"] < 0.05  else "ns"))
        print(f"  {s:50s}  Δ={cmp['mean_diff']:+.3f}  t_p={cmp['t_p']:.3e}  {stars}")

    # ── Plot ──
    plot_repeated_bars(df_results, outdir, n_classes=len(selected_labels))

    print(f"\n[Done] Results in: {outdir}")
    print("  repeat_results.csv          → per-seed train/val accuracy")
    print("  significance_summary.json   → all-pairs paired t-test, Wilcoxon, bootstrap CI")
    print("  probe_accuracy_repeated.png → mean ± SEM bar chart")


if __name__ == "__main__":
    main()
