"""
vision_patch_pairs_multiscale.py

Data pipeline for multi-scale spatial contrastive training.
For each anchor patch, samples one neighbor at each of the specified
adjacency distances (scales), producing:

  output: [B, S+1, D]
    dim 1:  [anchor, scale-1 neighbor, scale-2 neighbor, scale-4 neighbor]

Neighbor sampling at each scale:
  For scale d, randomly pick a patch that is exactly d steps away
  in Chebyshev distance (i.e., max(|dr|, |dc|) == d).
  If no valid neighbor exists at exactly distance d, falls back to
  the closest available distance.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], image_size: int = 224):
        self.image_paths = [str(p) for p in image_paths]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def _patches_at_chebyshev_distance(
    h: int, w: int, r: int, c: int, d: int
) -> List[tuple[int, int]]:
    """
    Return all patches at exactly Chebyshev distance d from (r, c).
    Chebyshev distance = max(|dr|, |dc|).
    """
    neighbors = []
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if max(abs(dr), abs(dc)) != d:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                neighbors.append((rr, cc))
    return neighbors


def _patches_at_most_chebyshev_distance(
    h: int, w: int, r: int, c: int, d: int
) -> List[tuple[int, int]]:
    """Fallback: return all patches within Chebyshev distance d."""
    neighbors = []
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                neighbors.append((rr, cc))
    return neighbors


class DINOFeatureExtractor:
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        import os
        os.environ["TORCH_HOME"] = "/home/ubuntu/.cache/torch"
        self.device = device
        self.model = torch.hub.load(
            "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
            model_name,
            source="local",
            trust_repo=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] → [B, N, D]"""
        images = images.to(self.device)
        feats = self.model.forward_features(images)
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"]
        raise RuntimeError("Cannot find x_norm_patchtokens.")


class MultiScalePatchPairBuffer:
    """
    Produces batches of shape [B_pairs, S+1, D]:
      dim 1:  [anchor, neighbor_scale1, neighbor_scale2, ...]

    Args:
        image_paths:      list of image file paths
        scales:           list of Chebyshev distances, e.g. [1, 2, 4]
        pairs_per_image:  number of anchor patches sampled per image
        dino_model_name:  DINOv2 model variant
        batch_size_images: images per forward pass through DINOv2
        image_size:       resize target
        device:           "cuda" or "cpu"
        shuffle:          whether to shuffle image order each epoch
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        scales: list[int] = [1, 2, 4],
        pairs_per_image: int = 64,
        dino_model_name: str = "dinov2_vitb14",
        batch_size_images: int = 16,
        image_size: int = 224,
        device: str = "cuda",
        shuffle: bool = True,
    ):
        self.scales = scales
        self.pairs_per_image = pairs_per_image
        self.device = device

        dataset = ImageFolderDataset(image_paths, image_size=image_size)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size_images,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        self.extractor = DINOFeatureExtractor(
            model_name=dino_model_name, device=device
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        for images in self.loader:
            with torch.no_grad():
                tokens = self.extractor.patch_tokens(images)  # [B, N, D]

            bsz, n_patches, d_model = tokens.shape
            side = int(math.sqrt(n_patches))
            if side * side != n_patches:
                continue

            grid = tokens.view(bsz, side, side, d_model)  # [B, H, W, D]

            sample_list = []
            for b in range(bsz):
                for _ in range(self.pairs_per_image):
                    r = random.randrange(side)
                    c = random.randrange(side)

                    anchor = grid[b, r, c]   # [D]

                    # For each scale, sample a neighbor
                    neighbors = []
                    valid = True
                    for d in self.scales:
                        nbrs = _patches_at_chebyshev_distance(side, side, r, c, d)
                        if not nbrs:
                            # Fallback to closest available distance
                            nbrs = _patches_at_most_chebyshev_distance(
                                side, side, r, c, d
                            )
                        if not nbrs:
                            valid = False
                            break
                        rr, cc = random.choice(nbrs)
                        neighbors.append(grid[b, rr, cc])  # [D]

                    if not valid:
                        continue

                    # Stack: [S+1, D]
                    sample = torch.stack([anchor] + neighbors, dim=0)
                    sample_list.append(sample)

            if not sample_list:
                continue

            # [B_pairs, S+1, D]
            x = torch.stack(sample_list, dim=0).to(self.device)
            yield x