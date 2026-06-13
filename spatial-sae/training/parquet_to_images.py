"""Convert Hugging Face parquet dataset files into local image folders.

Usage:
    python parquet_to_images.py \
        --parquet-files /path/train-00000-of-00001-*.parquet /path/valid-00000-of-00001-*.parquet \
        --out-dir ./imagenet_data

It will create one folder per class label and save each image as JPG.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from io import BytesIO

from datasets import load_dataset
from PIL import Image


def find_image_column(features: dict) -> str:
    # Prefer common image field names
    for preferred in ["image", "img", "pixel_values", "image_data"]:
        if preferred in features:
            return preferred

    # Exclude label-like fields from candidate list
    exclude = {
        "label",
        "labels",
        "class",
        "class_id",
        "category",
        "label_name",
        "label_text",
    }
    candidates = [name for name in features if name not in exclude]
    if not candidates:
        raise ValueError("Could not find an image column in the parquet dataset.")
    return candidates[0]


def get_label_value(sample: dict) -> str:
    if "label" in sample:
        return sample["label"]
    for candidate in ["class", "label_name", "category", "label_text"]:
        if candidate in sample:
            return sample[candidate]
    raise ValueError("Could not find a label field in the dataset sample.")


def image_from_sample(image_field):
    if image_field is None:
        raise ValueError("Image field is empty.")

    if hasattr(image_field, "to_pil"):
        return image_field.to_pil()

    if isinstance(image_field, dict):
        if "bytes" in image_field:
            return Image.open(BytesIO(image_field["bytes"])).convert("RGB")
        if "path" in image_field:
            return Image.open(image_field["path"]).convert("RGB")
        if "array" in image_field:
            return Image.fromarray(image_field["array"]).convert("RGB")

    if isinstance(image_field, (bytes, bytearray)):
        return Image.open(BytesIO(image_field)).convert("RGB")

    if isinstance(image_field, str):
        return Image.open(image_field).convert("RGB")

    if hasattr(image_field, "shape"):
        return Image.fromarray(image_field).convert("RGB")

    raise ValueError(f"Unsupported image field type: {type(image_field)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert parquet dataset to image folders.")
    parser.add_argument(
        "--parquet-files",
        nargs="+",
        required=True,
        help="Path(s) to parquet file(s).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./imagenet_data",
        help="Output directory for image folders.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional maximum images per class (0 = no limit).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    total = 0
    processed_any = False

    for file_path in args.parquet_files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Skip missing file: {file_path}")
            continue

        print(f"Processing parquet file: {file_path}")
        try:
            ds = load_dataset("parquet", data_files=str(file_path), split="train")
        except Exception as exc:
            print(f"  Warning: failed to load {file_path}: {exc}")
            continue

        processed_any = True
        features = ds.features
        image_col = find_image_column(features)
        print(f"  Detected image column: {image_col}")

        for idx, sample in enumerate(ds):
            label = get_label_value(sample)
            if isinstance(label, list):
                label = label[0]

            if args.max_per_class and counts.get(label, 0) >= args.max_per_class:
                continue

            image = image_from_sample(sample[image_col])

            label_dir = out_dir / str(label)
            label_dir.mkdir(parents=True, exist_ok=True)

            image_path = label_dir / f"{total:08d}.jpg"
            image.save(image_path, format="JPEG")

            counts[label] = counts.get(label, 0) + 1
            total += 1
            if total % 1000 == 0:
                print(f"  Saved {total} images...")

    if not processed_any:
        raise RuntimeError("No valid parquet files could be loaded.")

    print(f"Done. Saved {total} images into {out_dir}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
