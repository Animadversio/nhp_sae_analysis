"""
Step 1: 提取 DINOv2 spatial features（保留 patch 维度）
运行方式: python step1_extract_features.py --img_dir "C:\path\to\images" --device cuda
"""
import argparse, os, pickle
import torch, numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', required=True, help='NSD 图像文件夹路径')
parser.add_argument('--device',  default='cuda')
parser.add_argument('--block',   type=int, default=11)
parser.add_argument('--batch',   type=int, default=32)
args = parser.parse_args()

OUT_PATH = f'cache/dinov2_spatial_block{args.block}.pkl'
os.makedirs('cache', exist_ok=True)

print(f"Loading DINOv2 ViT-B/14-reg on {args.device}...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
model = model.to(args.device).eval()
model.requires_grad_(False)

N_REGISTERS = 4
tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

exts = {'.jpg', '.jpeg', '.png', '.bmp'}
img_paths = sorted([p for p in Path(args.img_dir).iterdir()
                    if p.suffix.lower() in exts])
N = len(img_paths)
print(f"Found {N} images in {args.img_dir}")
assert N > 0, "No images found. Check --img_dir."

patch_feats, cls_feats = [], []

def hook_fn(mod, inp, out):
    cls_feats.append(out[:, 0, :].detach().cpu())
    patch_feats.append(out[:, 1 + N_REGISTERS:, :].detach().cpu())

handle = model.blocks[args.block].register_forward_hook(hook_fn)

for start in range(0, N, args.batch):
    batch_paths = img_paths[start:start + args.batch]
    imgs = torch.stack([tfm(Image.open(p).convert('RGB')) for p in batch_paths])
    with torch.no_grad():
        model(imgs.to(args.device))
    print(f"  {start + len(batch_paths)}/{N}")

handle.remove()

patch_arr = np.concatenate([t.numpy() for t in patch_feats], axis=0)
cls_arr   = np.concatenate([t.numpy() for t in cls_feats],   axis=0)
print(f"patch_tokens: {patch_arr.shape}")
print(f"cls_tokens  : {cls_arr.shape}")

cache = {
    f'blocks.{args.block}_patch_spatial': patch_arr,
    f'blocks.{args.block}_cls':           cls_arr,
    'image_paths': [str(p) for p in img_paths],
}
with open(OUT_PATH, 'wb') as f:
    pickle.dump(cache, f)
print(f"Saved -> {OUT_PATH}")
