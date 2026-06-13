"""
analyze_highlevel_vs_lowlevel.py

Trains linear probes on each Matryoshka group subset (G0–G3) plus full features,
compares probe accuracy, and visualizes top discriminative concepts.

Usage:
    python -u scripts/analyze_highlevel_vs_lowlevel.py \\
        --parquet imagenet_val_probe.parquet \\
        --ckpt ckpts_multiscale_v1/ae_final.pt \\
        --n_classes 20 \\
        --n_images_per_class 50 \\
        --device cuda \\
        --outdir results_multiscale/highlevel_vs_lowlevel
"""

from __future__ import annotations

import argparse
import math
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
import json
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def minmax_norm(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ── Step 1: Compute latent statistics ────────────────────────

@torch.no_grad()
def compute_latent_stats(
    df, label_col, extractor, ae, device,
    image_size, dict_size, threshold=0.2, n_stat_images=1000,
):
    """Compute per-latent activated_frequency and mean_activation."""
    stat_df = df.sample(min(n_stat_images, len(df)), random_state=0)
    activation_count = np.zeros(dict_size)
    activation_sum   = np.zeros(dict_size)
    n = 0

    print(f"[Stats] Computing latent statistics on {len(stat_df)} images...")
    for _, row in stat_df.iterrows():
        try:
            tensor   = load_tensor(get_img_bytes(row), image_size)
            tokens   = extractor.patch_tokens(tensor)
            features = ae.encode(tokens.to(device), use_threshold=False).cpu()
            img_act  = features.mean(dim=0).numpy()
            active   = img_act > threshold
            activation_count += active.astype(float)
            activation_sum   += np.where(active, img_act, 0.0)
            n += 1
        except Exception:
            continue

    freq     = activation_count / max(n, 1)
    mean_act = np.where(activation_count > 0, activation_sum / activation_count, 0.0)
    return freq, mean_act


# ── Step 2: Split concepts into 4 groups ─────────────────────

def split_concepts(freq, mean_act, dict_size=16384, hl_fraction=0.25, **kwargs):
    """
    Split latents into groups by group_fractions (or equal hl_fraction splits).
    Returns list of index arrays (global indices) and group names.
    """
    group_fractions = kwargs.get("group_fractions", None)
    group_steps     = kwargs.get("group_steps", None)

    if group_fractions is not None:
        n_groups = len(group_fractions)
        boundaries = [0]
        for f in group_fractions[:-1]:
            boundaries.append(boundaries[-1] + int(dict_size * f))
        boundaries.append(dict_size)
    else:
        n_groups = 4
        group_size = int(dict_size * hl_fraction)
        boundaries = [g * group_size for g in range(n_groups)] + [dict_size]

    if group_steps is not None:
        group_names = [f"G{i} (step={s})" if s > 0 else f"G{i} (recon-only)"
                       for i, s in enumerate(group_steps)]
    elif group_fractions is not None:
        group_names = [f"G{i} ({int(f*100)}%)" for i, f in enumerate(group_fractions)]
    else:
        group_names = ["G0 (dist=4)", "G1 (dist=2)", "G2 (dist=1)", "G3 (recon-only)"]

    group_indices = []
    for g in range(n_groups):
        start, end = boundaries[g], boundaries[g + 1]
        idx = np.arange(start, end)
        print(f"[Split] {group_names[g]}: {len(idx)} features ({start}~{end-1})")
        group_indices.append(idx)
    return group_indices, group_names


# ── Step 3: Encode dataset ────────────────────────────────────

@torch.no_grad()
def encode_dataset(df, label_col, selected_labels, extractor, ae,
                   device, image_size, dict_size, n_images_per_class,
                   random_state=42):
    """Encode images → mean-pooled concept vectors [dict_size]."""
    label_to_idx = {str(l): i for i, l in enumerate(selected_labels)}
    X_list, y_list = [], []

    for label in selected_labels:
        rows = df[df[label_col] == label].sample(
            min(n_images_per_class, len(df[df[label_col] == label])),
            random_state=random_state,
        )
        for _, row in rows.iterrows():
            try:
                tensor   = load_tensor(get_img_bytes(row), image_size)
                tokens   = extractor.patch_tokens(tensor)
                features = ae.encode(tokens.to(device), use_threshold=False).cpu()
                vec      = features.mean(dim=0).numpy()
                X_list.append(vec)
                y_list.append(label_to_idx[str(label)])
            except Exception:
                continue
        print(f"  [{label}] encoded {sum(1 for y in y_list if y == label_to_idx[str(label)])} images")

    return np.stack(X_list), np.array(y_list), label_to_idx


@torch.no_grad()
def encode_dataset_raw(df, label_col, selected_labels, extractor,
                       device, image_size, n_images_per_class,
                       random_state=42):
    """Encode images → mean-pooled raw DINOv2 patch tokens [activation_dim]."""
    label_to_idx = {str(l): i for i, l in enumerate(selected_labels)}
    X_list, y_list = [], []

    for label in selected_labels:
        rows = df[df[label_col] == label].sample(
            min(n_images_per_class, len(df[df[label_col] == label])),
            random_state=random_state,
        )
        for _, row in rows.iterrows():
            try:
                tensor = load_tensor(get_img_bytes(row), image_size)
                tokens = extractor.patch_tokens(tensor)   # [P, D]
                vec    = tokens.mean(dim=0).cpu().numpy()
                X_list.append(vec)
                y_list.append(label_to_idx[str(label)])
            except Exception:
                continue

    return np.stack(X_list), np.array(y_list)


# ── Step 4: Train and evaluate probe ─────────────────────────

def train_and_evaluate_probe(X, y, concept_indices, subset_name, n_components=250, seed=42):
    """Train logistic regression probe on selected feature subset."""
    X_sub = X[:, concept_indices]

    # Split first to avoid data leakage
    X_train, X_val, y_train, y_val = train_test_split(
        X_sub, y, test_size=0.2, random_state=seed, stratify=y,
    )

    if n_components is not None and X_train.shape[1] > n_components:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_val   = pca.transform(X_val)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    clf = LogisticRegressionCV(
        Cs=[0.01, 0.1, 1.0, 10.0],
        cv=5, max_iter=1000, solver="saga", random_state=42,
    )
    clf.fit(X_train_s, y_train)
    print(f"  Best C: {clf.C_[0]:.4f}")

    train_acc = clf.score(X_train_s, y_train)
    val_acc   = clf.score(X_val_s,   y_val)

    print(f"\n[Probe: {subset_name}]")
    print(f"  Features used:   {len(concept_indices)}")
    print(f"  Train accuracy:  {train_acc:.3f}")
    print(f"  Val accuracy:    {val_acc:.3f}")
    print(f"  Random baseline: {1/len(np.unique(y)):.3f}")
    print(classification_report(y_val, clf.predict(X_val_s), zero_division=0))

    return clf, scaler, train_acc, val_acc


# ── Step 5: Accuracy comparison bar chart ────────────────────

def plot_accuracy_comparison(results, outdir, n_classes=20):
    """Bar chart with random chance baseline."""
    names      = list(results.keys())
    train_accs = [results[n][0] for n in names]
    val_accs   = [results[n][1] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(names)), 5))
    ax.bar(x - width/2, train_accs, width, label="Train accuracy", color="#2563EB", alpha=0.8)
    ax.bar(x + width/2, val_accs,   width, label="Val accuracy",   color="#16A34A", alpha=0.8)

    chance = 1.0 / n_classes
    ax.axhline(chance, linestyle="--", color="red", linewidth=1.5,
               label=f"Chance (1/{n_classes} = {chance:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Linear Probe Accuracy: Per-Group vs Full Features", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for i, (tr, vl) in enumerate(zip(train_accs, val_accs)):
        ax.text(i - width/2, tr + 0.01, f"{tr:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, vl + 0.01, f"{vl:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = outdir / "probe_accuracy_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


# ── Step 6: Visualize top discriminative concepts ────────────

@torch.no_grad()
def visualize_top_concepts(
    clf, concept_indices, subset_name,
    df, label_col, selected_labels, label_to_idx,
    extractor, ae, device, image_size,
    outdir, top_k_concepts=6,
):
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    clf_classes  = clf.classes_
    n_classes    = len(clf_classes)
    coef         = clf.coef_

    fig, axes = plt.subplots(
        n_classes, top_k_concepts + 1,
        figsize=(2.8 * (top_k_concepts + 1), 3.0 * n_classes),
        squeeze=False,
    )

    for coef_row_idx, class_idx in enumerate(clf_classes):
        label     = idx_to_label[class_idx]
        weights   = coef[coef_row_idx]
        top_local = np.argsort(np.abs(weights))[-top_k_concepts:][::-1]
        top_global = concept_indices[top_local]

        class_rows = df[df[label_col] == label]
        if len(class_rows) == 0:
            for ax in axes[coef_row_idx]:
                ax.axis("off")
            continue
        example_row = class_rows.sample(1, random_state=2).iloc[0]

        try:
            img_bytes = get_img_bytes(example_row)
            pil       = load_pil(img_bytes, image_size)
            tensor    = load_tensor(img_bytes, image_size)
            tokens    = extractor.patch_tokens(tensor)
            features  = ae.encode(tokens.to(device), use_threshold=False).cpu()
            side      = int(math.sqrt(tokens.shape[0]))

            axes[coef_row_idx, 0].imshow(pil)
            axes[coef_row_idx, 0].set_title(f"{label}", fontsize=7)
            axes[coef_row_idx, 0].axis("off")

            for col, (local_idx, global_idx) in enumerate(zip(top_local, top_global), start=1):
                fmap      = features[:, global_idx].view(side, side).numpy()
                fmap_norm = minmax_norm(fmap)
                w         = weights[local_idx]
                cmap      = "Reds" if w > 0 else "Blues"

                axes[coef_row_idx, col].imshow(pil)
                axes[coef_row_idx, col].imshow(
                    fmap_norm, alpha=0.6, interpolation="bilinear",
                    extent=(0, image_size, image_size, 0),
                    cmap=cmap, vmin=0, vmax=1,
                )
                axes[coef_row_idx, col].set_title(
                    f"C{global_idx}\n({'+' if w>0 else ''}{w:.2f})", fontsize=6,
                )
                axes[coef_row_idx, col].axis("off")

        except Exception:
            for col in range(top_k_concepts + 1):
                axes[coef_row_idx, col].axis("off")

    plt.suptitle(
        f"Top discriminative concepts [{subset_name}]\n"
        f"Red=positive indicator | Blue=negative indicator",
        fontsize=11,
    )
    plt.tight_layout()
    path = outdir / f"top_concepts_{subset_name.replace(' ', '_')}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Step 5b: Repeated-run statistics and plots ───────────────

def plot_repeated_bars(df_results, outdir, n_classes=20):
    """Bar plot of mean ± SEM validation accuracy across seeds."""
    import pandas as pd
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
    bars = ax.bar(x, means, width, yerr=sems, color="#2563EB", alpha=0.8,
                  capsize=4, label="Mean val accuracy ± SEM",
                  error_kw=dict(elinewidth=1.5))
    ax.axhline(chance, linestyle="--", color="red", linewidth=1.5,
               label=f"Chance (1/{n_classes} = {chance:.2f})")

    for i, (m, s) in enumerate(zip(means, sems)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Linear Probe Validation Accuracy (mean ± SEM across seeds)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = outdir / "probe_accuracy_repeated.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def compute_significance(df_results, reference="Raw DINO", n_bootstrap=5000, rng_seed=0):
    """
    All-pairs significance tests between every subset, plus per-subset summary stats.
    For each ordered pair (A, B): paired t-test, Wilcoxon, bootstrap 95% CI for mean(A-B).
    Returns a dict ready for JSON serialisation.
    """
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
        key = f"{a} vs {b}"
        summary["pairwise_comparisons"][key] = _pairwise(a, b)

    # Per-subset summary stats
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


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",            type=str, required=True)
    parser.add_argument("--ckpt",               type=str, required=True)
    parser.add_argument("--n_classes",          type=int, default=20)
    parser.add_argument("--n_images_per_class", type=int, default=50)
    parser.add_argument("--threshold",          type=float, default=0.2)
    parser.add_argument("--n_stat_images",      type=int, default=1000)
    parser.add_argument("--top_k_concepts",     type=int, default=6)
    parser.add_argument("--n_repeats",          type=int, default=20,
                        help="Number of repeated probe evaluations with different seeds.")
    parser.add_argument("--dino_model",  type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--dict_size",   type=int, default=16384)
    parser.add_argument("--k",           type=int, default=32)
    parser.add_argument("--hl_fraction",    type=float, default=0.25)
    parser.add_argument("--group_fractions", type=float, nargs="+", default=None,
                        help="group fractions used in training, e.g. --group_fractions 0.2 0.8")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--outdir",      type=str, default="results/highlevel_vs_lowlevel")
    parser.add_argument("--group_steps", type=int, nargs="+", default=None,
                        help="group_steps used in training, e.g. --group_steps 1 2 4 8")
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

    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    sample_tensor = load_tensor(get_img_bytes(df.iloc[0]), args.image_size)
    with torch.no_grad():
        activation_dim = extractor.patch_tokens(sample_tensor).shape[-1]

    print(f"[Model] Loading SAE from {args.ckpt}...")
    ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    # Compute latent stats on a fixed sample for group splitting
    freq, mean_act = compute_latent_stats(
        df=df_selected, label_col=label_col,
        extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, dict_size=args.dict_size,
        threshold=args.threshold, n_stat_images=args.n_stat_images,
    )
    group_indices_list, group_probe_names = split_concepts(
        freq, mean_act, dict_size=args.dict_size, hl_fraction=args.hl_fraction,
        group_fractions=args.group_fractions, group_steps=args.group_steps,
    )
    all_indices = np.arange(args.dict_size)
    N_DIM = 250

    # ── Multi-seed repeated evaluation ───────────────────────
    print(f"\n{'='*60}")
    print(f"REPEATED PROBE EVALUATION: {args.n_repeats} seeds")
    print(f"{'='*60}")

    all_rows = []   # list of dicts → CSV

    for seed in range(args.n_repeats):
        print(f"\n── Seed {seed}/{args.n_repeats - 1} ──")

        # Same image sample for SAE and raw DINO within this seed
        X, y, label_to_idx = encode_dataset(
            df=df_selected, label_col=label_col,
            selected_labels=selected_labels,
            extractor=extractor, ae=ae, device=args.device,
            image_size=args.image_size, dict_size=args.dict_size,
            n_images_per_class=args.n_images_per_class,
            random_state=seed,
        )
        X_raw, y_raw = encode_dataset_raw(
            df=df_selected, label_col=label_col,
            selected_labels=selected_labels,
            extractor=extractor, device=args.device,
            image_size=args.image_size,
            n_images_per_class=args.n_images_per_class,
            random_state=seed,
        )

        # Raw DINO baseline
        raw_indices = np.arange(X_raw.shape[1])
        _, _, tr_raw, vl_raw = train_and_evaluate_probe(
            X_raw, y_raw, raw_indices, "Raw DINO", n_components=N_DIM, seed=seed)
        all_rows.append({"seed": seed, "subset": "Raw DINO",
                         "train_acc": tr_raw, "val_acc": vl_raw})

        # SAE group probes
        for g_indices, g_name in zip(group_indices_list, group_probe_names):
            _, _, tr_g, vl_g = train_and_evaluate_probe(
                X, y, g_indices, g_name, n_components=N_DIM, seed=seed)
            all_rows.append({"seed": seed, "subset": g_name,
                             "train_acc": tr_g, "val_acc": vl_g})

        # Full SAE
        _, _, tr_all, vl_all = train_and_evaluate_probe(
            X, y, all_indices, "Full", n_components=N_DIM, seed=seed)
        all_rows.append({"seed": seed, "subset": "Full",
                         "train_acc": tr_all, "val_acc": vl_all})

    # ── Save results CSV ──────────────────────────────────────
    df_results = pd.DataFrame(all_rows)
    csv_path = outdir / "repeat_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Statistics ────────────────────────────────────────────
    sig_summary = compute_significance(df_results, reference="Raw DINO")
    json_path = outdir / "significance_summary.json"
    with open(json_path, "w") as f:
        json.dump(sig_summary, f, indent=2)
    print(f"Saved: {json_path}")

    # Print summary table
    print("\n── Validation accuracy summary ──")
    for s, st in sig_summary["subset_stats"].items():
        print(f"  {s:25s}  {st['mean']:.3f} ± {st['sem']:.3f}")
    print("\n── All-pairs significance ──")
    for s, cmp in sig_summary["pairwise_comparisons"].items():
        stars = "***" if cmp["t_p"] < 0.001 else ("**" if cmp["t_p"] < 0.01 else ("*" if cmp["t_p"] < 0.05 else "ns"))
        print(f"  {s:25s}  Δ={cmp['mean_diff']:+.3f}  t_p={cmp['t_p']:.3e}  {stars}")

    # ── Plots ─────────────────────────────────────────────────
    plot_repeated_bars(df_results, outdir, n_classes=len(selected_labels))

    # Also keep single-run comparison plot using last seed's results
    last_seed_results = {
        row["subset"]: (row["train_acc"], row["val_acc"])
        for row in all_rows if row["seed"] == args.n_repeats - 1
    }
    plot_accuracy_comparison(last_seed_results, outdir, n_classes=len(selected_labels))

    # ── Concept visualization (last seed's encodings) ────────
    print("\n[Viz] Visualizing top discriminative concepts (last seed)...")
    X_viz, y_viz, label_to_idx_viz = encode_dataset(
        df=df_selected, label_col=label_col,
        selected_labels=selected_labels,
        extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, dict_size=args.dict_size,
        n_images_per_class=args.n_images_per_class,
        random_state=args.n_repeats - 1,
    )
    g_first, g_last = 0, len(group_indices_list) - 1
    clf_g0, _, _, _ = train_and_evaluate_probe(
        X_viz, y_viz, group_indices_list[g_first], "G0_viz", n_components=N_DIM, seed=args.n_repeats - 1)
    clf_g3, _, _, _ = train_and_evaluate_probe(
        X_viz, y_viz, group_indices_list[g_last], "Glast_viz", n_components=N_DIM, seed=args.n_repeats - 1)

    for clf, indices, name in [
        (clf_g0, group_indices_list[g_first], f"G0_{group_probe_names[g_first].replace(' ', '_')}"),
        (clf_g3, group_indices_list[g_last],  f"Glast_{group_probe_names[g_last].replace(' ', '_')}"),
    ]:
        visualize_top_concepts(
            clf=clf, concept_indices=indices, subset_name=name,
            df=df_selected, label_col=label_col,
            selected_labels=selected_labels, label_to_idx=label_to_idx_viz,
            extractor=extractor, ae=ae, device=args.device,
            image_size=args.image_size, outdir=outdir,
            top_k_concepts=args.top_k_concepts,
        )

    print(f"\n[Done] Results saved to: {outdir}")
    print("  repeat_results.csv             → per-seed train/val accuracy")
    print("  significance_summary.json      → paired t-test, Wilcoxon, bootstrap CI")
    print("  probe_accuracy_repeated.png    → mean ± SEM bar plot across seeds")
    print("  probe_accuracy_comparison.png  → single-seed bar chart")
    print("  top_concepts_G0_dist4.png      → concepts from G0")
    print("  top_concepts_G3_recon_only.png → concepts from G3")


if __name__ == "__main__":
    main()
