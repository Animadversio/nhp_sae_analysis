"""
batch_visualize_concepts.py

批量可视化 SAE 学到的 concept，对多张图生成 top-k feature 的热力图。
同时统计哪些 feature 在整个数据集上最活跃，方便找到有意义的 concept。

用法:
    python batch_visualize_concepts.py \
        --image_root /path/to/images \
        --ckpt ./crt_05022100/ae_final.pt \
        --n_images 20 \
        --top_concepts 10 \
        --device cuda \
        --outdir ./concept_viz
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "temporal-saes"))

from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)


# ── 工具函数 ──────────────────────────────────────────────────────────────

class DINOFeatureExtractor:
    def __init__(self, model_name="dinov2_vitb14", device="cuda"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, image_tensor):
        feats = self.model.forward_features(image_tensor.to(self.device))
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"].squeeze(0)  # [N, D]
        raise RuntimeError("Cannot find x_norm_patchtokens.")


def load_image(path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    pil = Image.open(path).convert("RGB")
    tensor = transform(pil).unsqueeze(0)
    return pil, tensor


def build_sae(activation_dim, dict_size, k, device):
    group_fractions = [0.25, 0.25, 0.25, 0.25]
    group_sizes = [int(f * dict_size) for f in group_fractions[:-1]]
    group_sizes.append(dict_size - sum(group_sizes))
    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=dict_size,
        k=k,
        group_sizes=group_sizes,
    ).to(device)
    ae.eval()
    return ae


def minmax_norm(x):
    x_min, x_max = x.min(), x.max()
    if (x_max - x_min).abs() < 1e-8:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


# ── Phase 1: 统计全局最活跃的 concept ────────────────────────────────────

@torch.no_grad()
def compute_global_concept_strength(
    image_paths, extractor, ae, image_size, device, dict_size
):
    """
    遍历所有图片，统计每个 concept 的全局激活强度。
    返回 concept 强度向量 [dict_size]
    """
    global_strength = torch.zeros(dict_size)

    for path in image_paths:
        try:
            _, tensor = load_image(path, image_size)
            tokens = extractor.patch_tokens(tensor)          # [N, D]
            features = ae.encode(tokens.to(device)).cpu()    # [N, F]
            global_strength += features.sum(dim=0)           # 累加每个 concept 的总激活
        except Exception as e:
            print(f"  跳过 {path}: {e}")
            continue

    return global_strength


# ── Phase 2: 单图多 concept 可视化 ───────────────────────────────────────

@torch.no_grad()
def visualize_image_concepts(
    pil_image,
    tokens,
    ae,
    device,
    concept_indices,
    image_size,
    outdir,
    image_name,
):
    """
    对单张图，可视化指定 concept 列表的热力图。
    生成一张拼接图，每列是一个 concept。
    """
    side = int(math.sqrt(tokens.shape[0]))
    features = ae.encode(tokens.to(device)).cpu()  # [N, F]

    n_concepts = len(concept_indices)
    fig, axes = plt.subplots(2, n_concepts + 1, figsize=(3 * (n_concepts + 1), 7))

    # 第一列：原图
    img_resized = pil_image.resize((image_size, image_size))
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title("Original", fontsize=9)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # 每个 concept 一列
    for col, cidx in enumerate(concept_indices, start=1):
        fmap = features[:, cidx].view(side, side)
        fmap_norm = minmax_norm(fmap).numpy()

        # 上行：热力图
        im = axes[0, col].imshow(fmap_norm, interpolation="nearest", vmin=0, vmax=1)
        axes[0, col].set_title(f"C{cidx}", fontsize=8)
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # 下行：overlay
        axes[1, col].imshow(img_resized)
        axes[1, col].imshow(
            fmap_norm, alpha=0.55, interpolation="bilinear",
            extent=(0, image_size, image_size, 0), vmin=0, vmax=1
        )
        axes[1, col].set_title(f"C{cidx} overlay", fontsize=8)
        axes[1, col].axis("off")

    plt.suptitle(f"Top Concepts — {image_name}", fontsize=11)
    plt.tight_layout()

    save_path = outdir / f"{image_name}_concepts.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


# ── Phase 3: concept 跨图一致性可视化 ────────────────────────────────────

@torch.no_grad()
def visualize_concept_across_images(
    concept_idx,
    image_paths,
    extractor,
    ae,
    image_size,
    device,
    outdir,
    n_images=8,
):
    """
    对单个 concept，在多张图上展示它的激活热力图。
    用于判断这个 concept 在语义上是否一致。
    """
    fig, axes = plt.subplots(2, n_images, figsize=(3 * n_images, 7))

    shown = 0
    for path in image_paths:
        if shown >= n_images:
            break
        try:
            pil, tensor = load_image(path, image_size)
            tokens = extractor.patch_tokens(tensor)
            side = int(math.sqrt(tokens.shape[0]))
            features = ae.encode(tokens.to(device)).cpu()

            fmap = features[:, concept_idx].view(side, side)

            # 跳过这个 concept 几乎不激活的图
            if fmap.max() < 1e-4:
                continue

            fmap_norm = minmax_norm(fmap).numpy()
            img_resized = pil.resize((image_size, image_size))

            axes[0, shown].imshow(img_resized)
            axes[0, shown].set_title(f"img {shown}", fontsize=8)
            axes[0, shown].axis("off")

            axes[1, shown].imshow(img_resized)
            axes[1, shown].imshow(
                fmap_norm, alpha=0.6, interpolation="bilinear",
                extent=(0, image_size, image_size, 0), vmin=0, vmax=1
            )
            axes[1, shown].axis("off")

            shown += 1

        except Exception as e:
            continue

    if shown == 0:
        plt.close()
        return None

    plt.suptitle(f"Concept {concept_idx} across images", fontsize=12)
    plt.tight_layout()

    save_path = outdir / f"concept_{concept_idx}_across_images.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


# ── 主程序 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n_images", type=int, default=20,
                        help="用于可视化的图片数量")
    parser.add_argument("--top_concepts", type=int, default=10,
                        help="可视化最活跃的前 N 个 concept")
    parser.add_argument("--concepts_per_image", type=int, default=5,
                        help="每张图展示几个 concept")
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--dict_size", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--outdir", type=str, default="./concept_viz")
    parser.add_argument("--specific_concepts", type=int, nargs="+", default=None,
                        help="直接指定要可视化的 concept 编号，例如 --specific_concepts 11781 123 456")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 收集图片
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_paths = [str(p) for p in Path(args.image_root).rglob("*")
                 if p.suffix.lower() in exts]
    if not all_paths:
        raise ValueError(f"No images found in {args.image_root}")
    random.shuffle(all_paths)
    image_paths = all_paths[:args.n_images]
    print(f"使用 {len(image_paths)} 张图片")

    # 加载模型
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    # 推断 activation_dim
    _, tmp_tensor = load_image(image_paths[0], args.image_size)
    tmp_tokens = extractor.patch_tokens(tmp_tensor)
    activation_dim = tmp_tokens.shape[-1]

    # 加载 SAE
    ae = build_sae(activation_dim, args.dict_size, args.k, args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    ae.load_state_dict(state)
    ae.eval()
    print(f"SAE loaded from {args.ckpt}")

    # ── Phase 1: 找全局最活跃的 concept ──
    print("\n[Phase 1] 统计全局 concept 激活强度...")
    global_strength = compute_global_concept_strength(
        image_paths, extractor, ae, args.image_size, args.device, args.dict_size
    )
    top_vals, top_idx = torch.topk(global_strength, k=args.top_concepts)
    top_concept_indices = top_idx.tolist()

    print(f"\nTop {args.top_concepts} 最活跃 concept:")
    for rank, (idx, val) in enumerate(zip(top_concept_indices, top_vals.tolist())):
        print(f"  rank {rank:2d}: concept {idx:5d}, strength={val:.2f}")

    # ── Phase 2: 每张图的 top concept 热力图 ──
    print("\n[Phase 2] 生成每张图的 concept 热力图...")
    for path in image_paths[:10]:  # 最多处理10张
        try:
            pil, tensor = load_image(path, args.image_size)
            tokens = extractor.patch_tokens(tensor)
            image_name = Path(path).stem

            # 这张图自己最活跃的 concept
            with torch.no_grad():
                features = ae.encode(tokens.to(args.device)).cpu()
            img_strength = features.sum(dim=0)
            _, img_top_idx = torch.topk(img_strength, k=args.concepts_per_image)
            img_concept_indices = img_top_idx.tolist()

            save_path = visualize_image_concepts(
                pil, tokens, ae, args.device,
                img_concept_indices, args.image_size,
                outdir, image_name
            )
            print(f"  Saved: {save_path}")

        except Exception as e:
            print(f"  跳过 {path}: {e}")

    # ── Phase 3: 每个全局 top concept 跨图可视化 ──
    print(f"\n[Phase 3] 跨图可视化 top {args.top_concepts} concept...")
    for cidx in top_concept_indices:
        visualize_concept_across_images(
            concept_idx=cidx,
            image_paths=all_paths,   # 用全部图片找有激活的
            extractor=extractor,
            ae=ae,
            image_size=args.image_size,
            device=args.device,
            outdir=outdir,
            n_images=8,
        )

    print(f"\n完成！所有结果保存在: {outdir}")
    print("\n查看方式：")
    print(f"  - concept_XXXXX_across_images.png  →  单个 concept 在多图上的激活")
    print(f"  - IMAGENAME_concepts.png            →  单图上多个 concept 的热力图")


if __name__ == "__main__":
    main()