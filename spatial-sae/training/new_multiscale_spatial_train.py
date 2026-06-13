"""
new_multiscale_spatial_train.py
================================
Multi-scale Spatial SAE: four equal groups with per-group contrastive losses
at different spatial distances.

Group layout (group_steps controls which step-distance each group uses):
  G0: contrastive with N-step neighbor (most global, e.g. 3 or 4)
  G1: contrastive with (N-1)-step neighbor
  G2: contrastive with 1-step neighbor (most local)
  G3: reconstruction only, step=0 (pure low-level features)

Example --group_steps configurations:
  [3, 2, 1, 0]   default
  [4, 2, 1, 0]   log scale 4-2-1
  [8, 4, 2, 1]   log scale 8-4-2-1 (all groups have contrastive)

All other hyperparameters match the v4 training run.
"""

from __future__ import annotations

import argparse
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


# ─── Multi-step neighbor sampling ─────────────────────────────────────────────

def _neighbors_at_distance(h: int, w: int, r: int, c: int, d: int) -> list[tuple[int, int]]:
    """Return all grid positions at Manhattan distance exactly d from (r, c)."""
    out = []
    for dr in range(-d, d + 1):
        dc_abs = d - abs(dr)
        for dc in ([-dc_abs, dc_abs] if dc_abs > 0 else [0]):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _neighbors_within_distance(h: int, w: int, r: int, c: int, d: int) -> list[tuple[int, int]]:
    """Return all grid positions at Manhattan distance 1..d from (r, c)."""
    out = []
    for dist in range(1, d + 1):
        out.extend(_neighbors_at_distance(h, w, r, c, dist))
    return out


# ─── Multi-scale patch pair buffer ────────────────────────────────────────────

class MultiscalePatchPairBuffer:
    """
    Yield (B, n_slots, D) batches where:
      dim 0 = anchor patch
      dim 1..n_contrastive = neighbors at distances group_steps[i] for i where step > 0

    group_steps: list of int, one per group. 0 means no contrastive for that group.
    neighbor_mode: "exact" — picks neighbor at exactly that Manhattan distance.
                   "within" — picks neighbor within that distance (inclusive).
    """

    def __init__(
        self,
        image_paths: Optional[Sequence[Union[str, Path]]] = None,
        parquet_path: Optional[Union[str, Path]] = None,
        image_column: str = "image",
        dino_model_name: str = "dinov2_vitb14",
        dino_repo_path: Union[str, Path] = "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
        batch_size_images: int = 8,
        image_size: int = 224,
        pairs_per_image: int = 64,
        group_steps: List[int] = (3, 2, 1, 0),
        neighbor_sampling: str = "exact",
        device: str = "cuda",
        shuffle: bool = True,
        num_workers: int = 4,
        dino_block: Optional[int] = None,
    ):
        if parquet_path is None and image_paths is None:
            raise ValueError("Provide either parquet_path or image_paths.")
        if parquet_path is not None and image_paths is not None:
            raise ValueError("Provide only one of parquet_path or image_paths.")

        if parquet_path is not None:
            dataset = ParquetImageDataset(parquet_path, image_size=image_size, image_column=image_column)
        else:
            dataset = ImagePathDataset(image_paths or [], image_size=image_size)

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
            dino_repo_path=dino_repo_path,
            device=device,
            dino_block=dino_block,
        )
        self.pairs_per_image = pairs_per_image
        self.group_steps = list(group_steps)
        self.neighbor_sampling = neighbor_sampling
        self.device = device

        # Which groups have contrastive (step > 0)
        self.contrastive_groups = [(i, s) for i, s in enumerate(group_steps) if s > 0]
        # Number of neighbor slots = 1 (anchor) + number of contrastive groups
        self.n_slots = 1 + len(self.contrastive_groups)
        print(f"[MultiscalePatchPairBuffer] group_steps={group_steps}, "
              f"contrastive groups={self.contrastive_groups}, "
              f"batch slots={self.n_slots} (anchor + {len(self.contrastive_groups)} neighbors)")

    def __iter__(self) -> Iterator[torch.Tensor]:
        for images in self.loader:
            tokens = self.extractor.patch_tokens(images)  # (B, N, D)
            bsz, n_patches, d_model = tokens.shape
            side = int(math.sqrt(n_patches))
            assert side * side == n_patches, f"Expected square patch grid, got {n_patches} patches."
            grid = tokens.view(bsz, side, side, d_model)

            pair_list = []
            for b in range(bsz):
                for _ in range(self.pairs_per_image):
                    r = random.randrange(side)
                    c = random.randrange(side)

                    slots = [grid[b, r, c]]  # anchor

                    valid = True
                    for _, step_d in self.contrastive_groups:
                        if self.neighbor_sampling == "exact":
                            nbrs = _neighbors_at_distance(side, side, r, c, step_d)
                        else:  # "within"
                            nbrs = _neighbors_within_distance(side, side, r, c, step_d)
                        if not nbrs:
                            valid = False
                            break
                        rr, cc = random.choice(nbrs)
                        slots.append(grid[b, rr, cc])

                    if valid:
                        pair_list.append(torch.stack(slots, dim=0))  # (n_slots, D)

            if pair_list:
                yield torch.stack(pair_list, dim=0).to(self.device)  # (B, n_slots, D)


# ─── Multi-scale trainer (subclasses TemporalMatryoshkaBatchTopKTrainer) ──────

class MultiscaleSpatialTrainer(TemporalMatryoshkaBatchTopKTrainer):
    """
    Overrides the loss computation to apply per-group contrastive losses
    at different spatial distances. Uses raw dot-product contrastive (same as v4).
    """

    def __init__(self, group_steps: List[int], **kwargs):
        # Force temporal=True, contrastive=False (we handle contrastive ourselves)
        kwargs["temporal"] = True
        kwargs["contrastive"] = False
        super().__init__(**kwargs)
        self.group_steps = list(group_steps)
        self.contrastive_group_ids = [i for i, s in enumerate(group_steps) if s > 0]

    def loss(self, x: torch.Tensor, step: int, logging: bool = False):
        """
        x: (B, n_slots, D)
          x[:, 0] = anchor
          x[:, 1], x[:, 2], ... = neighbors for contrastive groups (in order)
        """
        import torch as t
        import torch.nn.functional as F

        # Encode anchor
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x[:, 0], return_active=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(f)

        # Split features and decoder by group
        W_dec_chunks = t.split(self.ae.W_dec, self.ae.group_sizes.tolist(), dim=0)
        f_chunks = t.split(f, self.ae.group_sizes.tolist(), dim=1)

        # Encode neighbors for each contrastive group
        # slot index: 1, 2, ... for contrastive_group_ids[0], [1], ...
        f_neighbor_by_group: dict[int, t.Tensor] = {}
        for slot_idx, grp_idx in enumerate(self.contrastive_group_ids):
            f_nbr, _, _ = self.ae.encode(
                x[:, slot_idx + 1], return_active=True, use_threshold=False
            )
            f_nbr_chunks = t.split(f_nbr, self.ae.group_sizes.tolist(), dim=1)
            f_neighbor_by_group[grp_idx] = f_nbr_chunks[grp_idx]

        # ── Reconstruction loss (Matryoshka cumulative) ──────────────────────
        x_reconstruct = t.zeros_like(x[:, 0]) + self.ae.b_dec
        total_l2_loss = t.tensor(0.0, device=self.device)
        for i in range(self.ae.active_groups):
            x_reconstruct = x_reconstruct + f_chunks[i] @ W_dec_chunks[i]
            l2 = (x[:, 0] - x_reconstruct).pow(2).sum(dim=-1).mean() * self.group_weights[i]
            total_l2_loss = total_l2_loss + l2

        # ── Per-group contrastive losses ─────────────────────────────────────
        total_contrastive = t.tensor(0.0, device=self.device)
        n_contrastive = len(self.contrastive_group_ids)
        for grp_idx, f_nbr in f_neighbor_by_group.items():
            logits = f_chunks[grp_idx] @ f_nbr.T  # raw dot-product (no normalization)
            labels = t.arange(logits.shape[0], device=self.device, dtype=t.long)
            cont = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            total_contrastive = total_contrastive + cont
        if n_contrastive > 0:
            total_contrastive = total_contrastive / n_contrastive  # average across groups

        # ── AuxK dead-feature loss ────────────────────────────────────────────
        auxk_loss = self.get_auxiliary_loss(
            (x[:, 0] - x_reconstruct).detach(), post_relu_acts_BF
        )

        loss = total_l2_loss + self.auxk_alpha * auxk_loss + self.temp_alpha * total_contrastive

        # Dead-feature tracking
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        if not logging:
            return loss
        else:
            from collections import namedtuple
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x[:, 0], x_reconstruct, f,
                {
                    "l2_loss": total_l2_loss.item(),
                    "contrastive_loss": total_contrastive.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                },
            )


# ─── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train multi-scale Spatial SAE with per-group contrastive losses."
    )

    # Data
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--image_root", type=str, default=None)
    p.add_argument("--image_column", type=str, default="image")

    # DINOv2
    p.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    p.add_argument(
        "--dino_repo_path", type=str,
        default="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache/hub/facebookresearch_dinov2_main",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--dino_block", type=int, default=None,
                   help="Hook intermediate DINOv2 block (1-indexed). Default: use final output.")

    # Patch pair sampling
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--batch_size_images", type=int, default=None)
    p.add_argument("--pairs_per_image", type=int, default=64)
    p.add_argument("--neighbor_sampling", type=str, default="exact",
                   choices=["exact", "within"],
                   help="'exact': pick neighbor at exactly the given Manhattan distance. "
                        "'within': pick neighbor anywhere within that distance.")
    p.add_argument("--num_workers", type=int, default=4)

    # Multi-scale contrastive: one step-distance per group (0 = no contrastive)
    p.add_argument("--group_steps", type=int, nargs="+", default=[3, 2, 1, 0],
                   help="Per-group contrastive step distances. e.g. '3 2 1 0' or '4 2 1 0' or '8 4 2 1'. "
                        "0 means reconstruction-only for that group.")

    # SAE / trainer — v4 defaults
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--dict_size", type=int, default=16384)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--group_fractions", type=float, nargs="+", default=[0.25, 0.25, 0.25, 0.25])
    p.add_argument("--group_weights", type=float, nargs="+", default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--auxk_alpha", type=float, default=1 / 32)
    p.add_argument("--temp_alpha", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--decay_start", type=int, default=None)
    p.add_argument("--threshold_beta", type=float, default=0.999)
    p.add_argument("--threshold_start_step", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)

    # Trainer metadata
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--lm_name", type=str, default="dinov2_vitb14")
    p.add_argument("--wandb_name", type=str, default="MultiscaleSpatialSAE")
    p.add_argument("--submodule_name", type=str, default="x_norm_patchtokens")

    # Checkpoint / runtime
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="results/multiscale_spatial_sae")
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--init_from_ckpt", type=str, default=None,
                   help="Warm-start from an existing ae state_dict (e.g. v4 ae_final.pt).")

    args = p.parse_args()
    if args.data_path is None and args.image_root is None:
        p.error("Provide either --data_path or --image_root.")
    if args.data_path is not None and args.image_root is not None:
        p.error("Provide only one of --data_path or --image_root.")
    return args


# ─── Helpers ───────────────────────────────────────────────────────────────────

def collect_image_paths(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise ValueError(f"No images found under: {root}")
    return paths


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size_images is None:
        args.batch_size_images = max(1, args.batch_size // max(1, args.pairs_per_image))

    # Validate group_steps vs group_fractions
    assert len(args.group_steps) == len(args.group_fractions), (
        f"--group_steps length ({len(args.group_steps)}) must match "
        f"--group_fractions length ({len(args.group_fractions)})"
    )

    image_paths = None
    parquet_path = None
    if args.data_path is not None:
        parquet_path = args.data_path
    else:
        image_paths = collect_image_paths(args.image_root)

    print(f"Multi-scale group_steps: {args.group_steps}")
    for i, s in enumerate(args.group_steps):
        label = f"contrastive @ dist={s}" if s > 0 else "reconstruction only"
        print(f"  G{i} → {label}")

    buffer = MultiscalePatchPairBuffer(
        image_paths=image_paths,
        parquet_path=parquet_path,
        image_column=args.image_column,
        dino_model_name=args.dino_model,
        dino_repo_path=args.dino_repo_path,
        batch_size_images=args.batch_size_images,
        image_size=args.image_size,
        pairs_per_image=args.pairs_per_image,
        group_steps=args.group_steps,
        neighbor_sampling=args.neighbor_sampling,
        device=args.device,
        shuffle=True,
        num_workers=args.num_workers,
        dino_block=args.dino_block,
    )

    # Infer activation_dim
    first_batch = next(iter(buffer))
    activation_dim = first_batch.shape[-1]
    print(f"activation_dim={activation_dim}, first_batch shape={tuple(first_batch.shape)}")

    trainer = MultiscaleSpatialTrainer(
        group_steps=args.group_steps,
        # TemporalMatryoshkaBatchTopKTrainer kwargs:
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

    if args.init_from_ckpt is not None:
        print(f"[Init] Loading weights from {args.init_from_ckpt}")
        state = torch.load(args.init_from_ckpt, map_location=args.device)
        trainer.ae.load_state_dict(state)
        print("[Init] Done.")

    # Save config
    with open(save_dir / "run_args.json", "w") as f:
        json.dump(vars(args) | {"activation_dim": activation_dim}, f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────
    step = 0
    while step < args.steps:
        for x in buffer:
            loss = trainer.update(step, x)
            if step % 10 == 0:
                print(f"[step {step}] loss={loss:.6f}")
            if step > 0 and step % args.save_every == 0:
                ckpt_path = save_dir / f"checkpoint_step_{step}.pt"
                torch.save(trainer.ae.state_dict(), ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")
            step += 1
            if step >= args.steps:
                break

    # Final checkpoint
    final_ckpt = save_dir / "ae_final.pt"
    torch.save(trainer.ae.state_dict(), final_ckpt)
    print(f"Saved final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
