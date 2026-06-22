"""
new_multiscale_spatial_train_v2.py
====================================
Multi-scale Spatial SAE with redesigned contrastive objectives:

  G0 – global semantic
       positive  : same image, Manhattan distance 6–8
       negative  : patches from OTHER images (implicit via InfoNCE batch)

  G1 – mid-level
       positive  : same image, distance 3–5
       hard-neg  : same image, distance 1–2   ← explicit hard negative
       cross-neg : other images               ← implicit via InfoNCE batch

  G2 – local
       positive  : same image, distance 1
       hard-neg  : same image, distance ≥ 4   ← explicit hard negative

  G3 – reconstruction only (no contrastive)

Batch format: (B, 6, D)
  slot 0 : anchor
  slot 1 : G0_pos   (same image, dist 6-8)
  slot 2 : G1_pos   (same image, dist 3-5)
  slot 3 : G1_neg   (same image, dist 1-2)  ← hard neg
  slot 4 : G2_pos   (same image, dist 1)
  slot 5 : G2_neg   (same image, dist >= 4) ← hard neg

InfoNCE losses:
  G0 : f0_anchor @ f0_G0pos.T        →  diagonal = positive, off-diagonal = cross-image
  G1 : f1_anchor @ [f1_G1pos; f1_G1neg].T  →  positive at diagonal of first B cols,
       hard-neg at last B cols (same image close neighbors + cross-image clutter)
  G2 : f2_anchor @ [f2_G2pos; f2_G2neg].T  →  positive at diagonal, hard-neg at last B cols
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
REPO_ROOT = SCRIPTS_DIR.parent
for p in [SCRIPTS_DIR, REPO_ROOT, REPO_ROOT / "temporal-saes" / "dictionary_learning"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from new_vision_patch_pairs import (  # noqa: E402
    DINOFeatureExtractor,
    ImagePathDataset,
    NSDHdf5Dataset,
    ParquetImageDataset,
)
from dictionary_learning.trainers.temporal_sequence_top_k import (  # noqa: E402
    TemporalMatryoshkaBatchTopKSAE,
    TemporalMatryoshkaBatchTopKTrainer,
)
from dictionary_learning.trainers.trainer import (  # noqa: E402
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)
from torch.utils.data import DataLoader


# ─── Distance helpers ──────────────────────────────────────────────────────────

def _neighbors_in_range(h: int, w: int, r: int, c: int,
                         d_min: int, d_max: int) -> list[tuple[int, int]]:
    """All grid positions at Manhattan distance d_min..d_max from (r, c)."""
    out = []
    for dr in range(-d_max, d_max + 1):
        for dc in range(-d_max, d_max + 1):
            dist = abs(dr) + abs(dc)
            if d_min <= dist <= d_max:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.append((rr, cc))
    return out


def _neighbors_at_least(h: int, w: int, r: int, c: int,
                          d_min: int) -> list[tuple[int, int]]:
    """All grid positions at Manhattan distance >= d_min from (r, c)."""
    out = []
    for rr in range(h):
        for cc in range(w):
            if abs(rr - r) + abs(cc - c) >= d_min:
                out.append((rr, cc))
    return out


# ─── New batch buffer ──────────────────────────────────────────────────────────

N_SLOTS = 6  # anchor + 5 neighbor slots

class MultiscalePatchPairBufferV2:
    """
    Yield (B, 6, D) batches:
      slot 0: anchor            (image i, patch (r,c))
      slot 1: G0_pos            (image i, dist 6-8)
      slot 2: G1_pos            (image i, dist 3-5)
      slot 3: G1_neg            (image i, dist 1-2)
      slot 4: G2_pos            (image i, dist 1)
      slot 5: G2_neg            (image i, dist >= 4)
    """

    def __init__(
        self,
        parquet_path: Optional[Union[str, Path]] = None,
        hdf5_path: Optional[Union[str, Path]] = None,
        hdf5_key: str = "imgBrick",
        image_paths: Optional[Sequence[Union[str, Path]]] = None,
        image_column: str = "image",
        dino_model_name: str = "dinov2_vitb14",
        dino_repo_path: Union[str, Path] = (
            "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/"
            "torch_cache/hub/facebookresearch_dinov2_main"
        ),
        batch_size_images: int = 8,
        image_size: int = 224,
        pairs_per_image: int = 64,
        device: str = "cuda",
        shuffle: bool = True,
        num_workers: int = 4,
        dino_block: Optional[int] = None,
        # Distance ranges
        g0_dist_min: int = 6,
        g0_dist_max: int = 8,
        g1_pos_dist_min: int = 3,
        g1_pos_dist_max: int = 5,
        g1_neg_dist_min: int = 1,
        g1_neg_dist_max: int = 2,
        g2_pos_dist: int = 1,
        g2_neg_dist_min: int = 4,
    ):
        if parquet_path is not None:
            dataset = ParquetImageDataset(parquet_path, image_size=image_size,
                                           image_column=image_column)
        elif hdf5_path is not None:
            dataset = NSDHdf5Dataset(hdf5_path, image_size=image_size, key=hdf5_key)
            print(f"[BufferV2] Using NSD HDF5 dataset: {hdf5_path} ({len(dataset)} images)")
        elif image_paths is not None:
            dataset = ImagePathDataset(image_paths, image_size=image_size)
        else:
            raise ValueError("Provide either parquet_path, hdf5_path, or image_paths.")

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size_images,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
            drop_last=False,
        )
        self.extractor = DINOFeatureExtractor(
            model_name=dino_model_name,
            dino_repo_path=str(dino_repo_path),
            device=device,
            dino_block=dino_block,
        )
        self.pairs_per_image = pairs_per_image
        self.device = device
        self.g0_dist_min = g0_dist_min
        self.g0_dist_max = g0_dist_max
        self.g1_pos_dist_min = g1_pos_dist_min
        self.g1_pos_dist_max = g1_pos_dist_max
        self.g1_neg_dist_min = g1_neg_dist_min
        self.g1_neg_dist_max = g1_neg_dist_max
        self.g2_pos_dist = g2_pos_dist
        self.g2_neg_dist_min = g2_neg_dist_min

        print(f"[BufferV2] G0_pos=dist({g0_dist_min}-{g0_dist_max}), "
              f"G1_pos=dist({g1_pos_dist_min}-{g1_pos_dist_max}), "
              f"G1_neg=dist({g1_neg_dist_min}-{g1_neg_dist_max}), "
              f"G2_pos=dist({g2_pos_dist}), "
              f"G2_neg=dist>={g2_neg_dist_min}), "
              f"n_slots={N_SLOTS}")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for images in self.loader:
            tokens = self.extractor.patch_tokens(images)   # (B_img, N, D)
            bsz, n_patches, d_model = tokens.shape
            side = int(math.sqrt(n_patches))
            assert side * side == n_patches
            grid = tokens.view(bsz, side, side, d_model)

            pair_list: list[torch.Tensor] = []
            img_idx_list: list[int] = []

            for b in range(bsz):
                for _ in range(self.pairs_per_image):
                    r = random.randrange(side)
                    c = random.randrange(side)

                    # Candidate neighbor sets
                    g0_pos_cands = _neighbors_in_range(side, side, r, c,
                                                        self.g0_dist_min, self.g0_dist_max)
                    g1_pos_cands = _neighbors_in_range(side, side, r, c,
                                                        self.g1_pos_dist_min, self.g1_pos_dist_max)
                    g1_neg_cands = _neighbors_in_range(side, side, r, c,
                                                        self.g1_neg_dist_min, self.g1_neg_dist_max)
                    g2_pos_cands = _neighbors_in_range(side, side, r, c,
                                                        self.g2_pos_dist, self.g2_pos_dist)
                    g2_neg_cands = _neighbors_at_least(side, side, r, c, self.g2_neg_dist_min)

                    # Skip anchor positions with no valid neighbors for any slot
                    if not (g0_pos_cands and g1_pos_cands and g1_neg_cands
                            and g2_pos_cands and g2_neg_cands):
                        continue

                    rr0, cc0 = random.choice(g0_pos_cands)
                    rr1p, cc1p = random.choice(g1_pos_cands)
                    rr1n, cc1n = random.choice(g1_neg_cands)
                    rr2p, cc2p = random.choice(g2_pos_cands)
                    rr2n, cc2n = random.choice(g2_neg_cands)

                    slots = torch.stack([
                        grid[b, r,   c],    # slot 0: anchor
                        grid[b, rr0, cc0],  # slot 1: G0_pos
                        grid[b, rr1p, cc1p],# slot 2: G1_pos
                        grid[b, rr1n, cc1n],# slot 3: G1_neg (hard)
                        grid[b, rr2p, cc2p],# slot 4: G2_pos
                        grid[b, rr2n, cc2n],# slot 5: G2_neg (hard)
                    ], dim=0)               # (6, D)
                    pair_list.append(slots)
                    img_idx_list.append(b)

            if pair_list:
                # Shuffle so pairs from different images are interleaved.
                # This ensures G0 InfoNCE off-diagonal elements are cross-image negatives,
                # not same-image patches grouped together.
                perm = list(range(len(pair_list)))
                random.shuffle(perm)
                pair_list    = [pair_list[i]    for i in perm]
                img_idx_list = [img_idx_list[i] for i in perm]

                tokens_out  = torch.stack(pair_list, dim=0).to(self.device)  # (B, 6, D)
                img_idx_out = torch.tensor(img_idx_list, device=self.device)  # (B,)
                yield tokens_out, img_idx_out


# ─── Trainer ───────────────────────────────────────────────────────────────────

class MultiscaleSpatialTrainerV2(TemporalMatryoshkaBatchTopKTrainer):
    """
    Loss:
      G0: InfoNCE (anchor vs G0_pos; same-image pairs masked out of negative pool)
      G1: InfoNCE with explicit hard-neg (G1_neg concat'd to negative pool)
      G2: InfoNCE with explicit hard-neg (G2_neg concat'd to negative pool)
      G3: reconstruction only
      All groups: Matryoshka reconstruction + AuxK
    """

    def __init__(self, **kwargs):
        kwargs["temporal"] = True
        kwargs["contrastive"] = False
        super().__init__(**kwargs)

    def loss(self, x, step: int, logging: bool = False):
        """x: (B, 6, D) or tuple((B, 6, D), (B,) img_idx)"""
        import torch as t

        if isinstance(x, (tuple, list)):
            x, img_idx = x   # img_idx: (B,) int tensor, same value = same source image
        else:
            img_idx = None

        anchor  = x[:, 0]   # (B, D)
        g0_pos  = x[:, 1]
        g1_pos  = x[:, 2]
        g1_neg  = x[:, 3]
        g2_pos  = x[:, 4]
        g2_neg  = x[:, 5]

        # ── Encode anchor ────────────────────────────────────────────────────
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            anchor, return_active=True, use_threshold=False
        )
        if step > self.threshold_start_step:
            self.update_threshold(f)

        group_sizes = self.ae.group_sizes.tolist()
        W_dec_chunks = t.split(self.ae.W_dec, group_sizes, dim=0)
        f_chunks     = t.split(f, group_sizes, dim=1)            # [f0, f1, f2, f3]

        # ── Encode neighbor slots ────────────────────────────────────────────
        def _enc_grp(tok: t.Tensor, g: int) -> t.Tensor:
            """Encode tokens, return only group g's features."""
            fe = self.ae.encode(tok, use_threshold=False)
            return t.split(fe, group_sizes, dim=1)[g]

        with t.no_grad():
            f0_pos = _enc_grp(g0_pos, 0)   # G0 pos features
            f1_pos = _enc_grp(g1_pos, 1)   # G1 pos features
            f1_neg = _enc_grp(g1_neg, 1)   # G1 neg features (hard)
            f2_pos = _enc_grp(g2_pos, 2)   # G2 pos features
            f2_neg = _enc_grp(g2_neg, 2)   # G2 neg features (hard)

        # ── Reconstruction loss (Matryoshka cumulative) ──────────────────────
        x_hat = t.zeros_like(anchor) + self.ae.b_dec
        total_l2 = t.tensor(0.0, device=self.device)
        for i in range(self.ae.active_groups):
            x_hat = x_hat + f_chunks[i] @ W_dec_chunks[i]
            l2 = (anchor - x_hat).pow(2).sum(-1).mean() * self.group_weights[i]
            total_l2 = total_l2 + l2

        # ── G0 InfoNCE: positive = G0_pos, negatives = cross-image only ─────
        # logits (B, B): diagonal = positive pair, off-diagonal should be cross-image neg
        logits0 = f_chunks[0] @ f0_pos.T
        if img_idx is not None:
            # Mask out same-image off-diagonal entries so they are NOT treated as negatives.
            # Two positions are same-image when img_idx[i] == img_idx[j] and i != j.
            same_img_mask = img_idx.unsqueeze(1) == img_idx.unsqueeze(0)  # (B, B)
            same_img_mask.fill_diagonal_(False)  # keep diagonal (positive pairs)
            logits0 = logits0.masked_fill(same_img_mask, float('-inf'))
        lbl = t.arange(logits0.shape[0], device=self.device, dtype=t.long)
        loss_g0 = (F.cross_entropy(logits0, lbl) + F.cross_entropy(logits0.T, lbl)) / 2

        # ── G1 InfoNCE with hard negatives ─────────────────────────────────
        # Negative pool: [G1_pos from OTHER images (B cols)] + [G1_neg hard (B cols)]
        # logits (B, 2B): positive at diagonal of first B cols
        neg_pool1 = t.cat([f1_pos, f1_neg], dim=0)        # (2B, D_g1)
        logits1   = f_chunks[1] @ neg_pool1.T              # (B, 2B)
        # positive index = i (diagonal of first B cols)
        loss_g1 = F.cross_entropy(logits1, lbl)

        # ── G2 InfoNCE with hard negatives ─────────────────────────────────
        neg_pool2 = t.cat([f2_pos, f2_neg], dim=0)        # (2B, D_g2)
        logits2   = f_chunks[2] @ neg_pool2.T              # (B, 2B)
        loss_g2 = F.cross_entropy(logits2, lbl)

        # G3 has no contrastive loss

        # ── AuxK loss ────────────────────────────────────────────────────────
        auxk_loss = self.get_auxiliary_loss(
            (anchor - x_hat).detach(), post_relu_acts_BF
        )

        contrastive = (loss_g0 + loss_g1 + loss_g2) / 3.0
        loss = total_l2 + self.auxk_alpha * auxk_loss + self.temp_alpha * contrastive

        # Dead-feature tracking
        num_tokens = anchor.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens
        self.num_tokens_since_fired[did_fire] = 0

        # Always store latest per-group losses for external CSV logging
        self._last_per_group_losses = {
            "l2_loss":          total_l2.item(),
            "loss_g0":          loss_g0.item(),
            "loss_g1":          loss_g1.item(),
            "loss_g2":          loss_g2.item(),
            "contrastive_loss": contrastive.item(),
            "auxk_loss":        auxk_loss.item(),
            "loss":             loss.item(),
        }

        if not logging:
            return loss
        from collections import namedtuple
        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
            anchor, x_hat, f,
            {
                "l2_loss":          total_l2.item(),
                "loss_g0":          loss_g0.item(),
                "loss_g1":          loss_g1.item(),
                "loss_g2":          loss_g2.item(),
                "contrastive_loss": contrastive.item(),
                "auxk_loss":        auxk_loss.item(),
                "loss":             loss.item(),
            },
        )


# ─── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",    type=str, default=None)
    p.add_argument("--hdf5_path",    type=str, default=None,
        help="Path to NSD-style HDF5 with key 'imgBrick' (N,H,W,3) uint8.")
    p.add_argument("--hdf5_key",     type=str, default="imgBrick")
    p.add_argument("--image_root",   type=str, default=None)
    p.add_argument("--image_column", type=str, default="image")
    p.add_argument("--dino_model",   type=str, default="dinov2_vitb14")
    p.add_argument("--dino_repo_path", type=str,
        default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache/hub/facebookresearch_dinov2_main")
    p.add_argument("--image_size",   type=int, default=224)
    p.add_argument("--dino_block",   type=int, default=None,
        help="Hook DINOv2 intermediate block (default: final output)")

    # Distance ranges (can override defaults)
    p.add_argument("--g0_dist_min",     type=int, default=6)
    p.add_argument("--g0_dist_max",     type=int, default=8)
    p.add_argument("--g1_pos_dist_min", type=int, default=3)
    p.add_argument("--g1_pos_dist_max", type=int, default=5)
    p.add_argument("--g1_neg_dist_min", type=int, default=1)
    p.add_argument("--g1_neg_dist_max", type=int, default=2)
    p.add_argument("--g2_pos_dist",     type=int, default=1)
    p.add_argument("--g2_neg_dist_min", type=int, default=4)

    p.add_argument("--batch_size",         type=int, default=512)
    p.add_argument("--batch_size_images",  type=int, default=None)
    p.add_argument("--pairs_per_image",    type=int, default=64)
    p.add_argument("--num_workers",        type=int, default=4)

    p.add_argument("--steps",           type=int,   default=30000)
    p.add_argument("--dict_size",       type=int,   default=5000)
    p.add_argument("--k",               type=int,   default=32)
    p.add_argument("--group_fractions", type=float, nargs="+", default=[0.25, 0.25, 0.25, 0.25])
    p.add_argument("--group_weights",   type=float, nargs="+", default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--auxk_alpha",      type=float, default=1/32)
    p.add_argument("--temp_alpha",      type=float, default=0.1)
    p.add_argument("--warmup_steps",    type=int,   default=1000)
    p.add_argument("--decay_start",     type=int,   default=None)
    p.add_argument("--threshold_beta",  type=float, default=0.999)
    p.add_argument("--threshold_start_step", type=int, default=1000)
    p.add_argument("--seed",            type=int,   default=0)

    p.add_argument("--layer",          type=int, default=0)
    p.add_argument("--lm_name",        type=str, default="dinov2_vitb14")
    p.add_argument("--wandb_name",     type=str, default="MultiscaleSpatialSAE_v2")
    p.add_argument("--submodule_name", type=str, default="x_norm_patchtokens")

    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir",    type=str, default="results/multiscale_v2")
    p.add_argument("--save_every",  type=int, default=10000)
    p.add_argument("--init_from_ckpt", type=str, default=None)

    p.add_argument("--log_every",    type=int, default=200,
        help="Log per-group losses to CSV every N steps.")

    args = p.parse_args()
    if args.data_path is None and args.hdf5_path is None and args.image_root is None:
        p.error("Provide --data_path, --hdf5_path, or --image_root.")
    return args


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size_images is None:
        args.batch_size_images = max(1, args.batch_size // max(1, args.pairs_per_image))

    print("=== MultiscaleSpatialSAE v2 ===")
    print(f"  G0: positive=dist({args.g0_dist_min}-{args.g0_dist_max}), negative=cross-image")
    print(f"  G1: positive=dist({args.g1_pos_dist_min}-{args.g1_pos_dist_max}), "
          f"hard-neg=dist({args.g1_neg_dist_min}-{args.g1_neg_dist_max}) + cross-image")
    print(f"  G2: positive=dist({args.g2_pos_dist}), "
          f"hard-neg=dist>={args.g2_neg_dist_min}")
    print(f"  G3: reconstruction only")
    print(f"  dict_size={args.dict_size}, k={args.k}, steps={args.steps}")

    parquet_path  = args.data_path
    hdf5_path     = args.hdf5_path
    image_paths   = None
    if args.data_path is None and args.hdf5_path is None:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = [str(p) for p in Path(args.image_root).rglob("*")
                       if p.suffix.lower() in exts]

    buffer = MultiscalePatchPairBufferV2(
        parquet_path=parquet_path,
        hdf5_path=hdf5_path,
        hdf5_key=args.hdf5_key,
        image_paths=image_paths,
        image_column=args.image_column,
        dino_model_name=args.dino_model,
        dino_repo_path=args.dino_repo_path,
        batch_size_images=args.batch_size_images,
        image_size=args.image_size,
        pairs_per_image=args.pairs_per_image,
        device=args.device,
        shuffle=True,
        num_workers=args.num_workers,
        dino_block=args.dino_block,
        g0_dist_min=args.g0_dist_min,
        g0_dist_max=args.g0_dist_max,
        g1_pos_dist_min=args.g1_pos_dist_min,
        g1_pos_dist_max=args.g1_pos_dist_max,
        g1_neg_dist_min=args.g1_neg_dist_min,
        g1_neg_dist_max=args.g1_neg_dist_max,
        g2_pos_dist=args.g2_pos_dist,
        g2_neg_dist_min=args.g2_neg_dist_min,
    )

    first_batch, _ = next(iter(buffer))
    activation_dim = first_batch.shape[-1]
    print(f"activation_dim={activation_dim}, first_batch={tuple(first_batch.shape)}")

    trainer = MultiscaleSpatialTrainerV2(
        steps=args.steps,
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        layer=args.layer,
        lm_name=args.lm_name,
        group_fractions=args.group_fractions,
        group_weights=args.group_weights,
        lr=args.lr,
        auxk_alpha=args.auxk_alpha,
        temp_alpha=args.temp_alpha,
        warmup_steps=args.warmup_steps,
        decay_start=args.decay_start,
        threshold_beta=args.threshold_beta,
        threshold_start_step=args.threshold_start_step,
        seed=args.seed,
        device=args.device,
        wandb_name=args.wandb_name,
        submodule_name=args.submodule_name,
        neighbor_recon=False,
        normalize_contrastive=False,
    )

    if args.init_from_ckpt:
        print(f"[Init] Loading weights from {args.init_from_ckpt}")
        state = torch.load(args.init_from_ckpt, map_location=args.device)
        trainer.ae.load_state_dict(state)

    with open(save_dir / "run_args.json", "w") as f:
        json.dump(vars(args) | {"activation_dim": activation_dim}, f, indent=2)

    loss_csv_path = save_dir / "loss_log.csv"
    csv_fields = ["step", "loss", "l2_loss", "loss_g0", "loss_g1", "loss_g2",
                  "contrastive_loss", "auxk_loss"]
    loss_records: list[dict] = []

    # Open CSV and write header
    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    step = 0
    while step < args.steps:
        for x in buffer:
            loss_val = trainer.update(step, x)
            if step % 100 == 0:
                print(f"[step {step:>6}] loss={loss_val:.6f}")

            # Log per-group losses every log_every steps
            if step % args.log_every == 0 and hasattr(trainer, "_last_per_group_losses"):
                row = {"step": step, **trainer._last_per_group_losses}
                loss_records.append(row)
                with open(loss_csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    writer.writerow(row)
                print(f"[step {step:>6}] loss_g0={row['loss_g0']:.4f} "
                      f"loss_g1={row['loss_g1']:.4f} loss_g2={row['loss_g2']:.4f} "
                      f"l2={row['l2_loss']:.4f}")

            if step > 0 and step % args.save_every == 0:
                ckpt = save_dir / f"checkpoint_step_{step}.pt"
                torch.save(trainer.ae.state_dict(), ckpt)
                print(f"Checkpoint: {ckpt}")
            step += 1
            if step >= args.steps:
                break

    final = save_dir / "ae_final.pt"
    torch.save(trainer.ae.state_dict(), final)
    print(f"Saved: {final}")
    print(f"Loss log: {loss_csv_path} ({len(loss_records)} entries)")


if __name__ == "__main__":
    main()
