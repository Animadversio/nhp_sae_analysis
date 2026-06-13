import sys, os
sys.path.insert(0, '/n/home12/binxuwang/T-SAE-Spatial/temporal-saes/dictionary_learning')
import torch
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from torchvision import transforms

os.environ['TORCH_HOME'] = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/torch_cache'

from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE

# Load SAE
ckpt = torch.load('/n/home12/binxuwang/T-SAE-Spatial/ckpts_multiscale_v1/ae_final.pt', map_location='cpu')
sd = ckpt['ae_state_dict'] if 'ae_state_dict' in ckpt else ckpt
group_sizes = sd['group_sizes'].tolist()
ae = TemporalMatryoshkaBatchTopKSAE(768, 16384, 32, group_sizes, temporal=False)
ae.load_state_dict(sd, strict=False)
ae = ae.cuda().eval()
print(f'Multiscale SAE: threshold={ae.threshold.item():.4f}, k={ae.k.item()}')

# Load DINOv2
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True).cuda().eval()
transform = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

df = pd.read_parquet('/n/home12/binxuwang/T-SAE-Spatial/data/imagenet_data/train-00000-of-00001-18bc3231d015f1e8.parquet').head(16)

imgs = []
for _, row in df.iterrows():
    img = Image.open(BytesIO(row['image']['bytes'])).convert('RGB')
    imgs.append(transform(img))

x_batch = torch.stack(imgs).cuda()
with torch.no_grad():
    out = dino.forward_features(x_batch)
    tokens = out['x_norm_patchtokens']  # (16, 256, 768)
    
    print(f'\nDINOv2 token stats: min={tokens.min().item():.3f} max={tokens.max().item():.3f} mean={tokens.mean().item():.3f}')

    # Encode single image (256 patches)
    single = tokens[0].cuda()  # (256, 768)
    feats_single = ae.encode(single, use_threshold=True)
    print(f'\nSingle image (256 patches):')
    print(f'  Nonzero per patch: {(feats_single!=0).float().sum(1).mean().item():.1f}')
    img_mean = feats_single.mean(0).cpu().numpy()
    print(f'  Features with mean>0.2: {(img_mean>0.2).sum()}')
    print(f'  Max mean activation: {img_mean.max():.4f}')
    
    # Encode 8 images together (2048 patches)
    batch8 = tokens[:8].reshape(-1, 768).cuda()  # (2048, 768)
    feats_batch = ae.encode(batch8, use_threshold=True)
    print(f'\n8 images together (2048 patches):')
    print(f'  Nonzero per patch: {(feats_batch!=0).float().sum(1).mean().item():.1f}')
    n_active_any_img = 0
    for i in range(8):
        img_f = feats_batch[i*256:(i+1)*256]
        img_m = img_f.mean(0).cpu().numpy()
        n_active_any_img += (img_m > 0.2).sum()
    print(f'  Total (feature, image) pairs active (sum over 8 imgs): {n_active_any_img}')
    print(f'  Avg features active per image: {n_active_any_img/8:.1f}')
    
    # Try use_threshold=False (topk)
    feats_topk = ae.encode(batch8, use_threshold=False)
    print(f'\n8 images topk (use_threshold=False):')
    print(f'  Nonzero per patch: {(feats_topk!=0).float().sum(1).mean().item():.1f}')
    n_active_topk = 0
    for i in range(8):
        img_f = feats_topk[i*256:(i+1)*256]
        img_m = img_f.mean(0).cpu().numpy()
        n_active_topk += (img_m > 0.01).sum()
    print(f'  Avg features with mean>0.01 per image: {n_active_topk/8:.1f}')
    
    # What threshold would give ~100 active features per image?
    feats_th = ae.encode(batch8, use_threshold=True)
    for thresh in [0.05, 0.1, 0.2, 0.5, 1.0]:
        n_act = sum(
            (feats_th[i*256:(i+1)*256].mean(0).cpu().numpy() > thresh).sum()
            for i in range(8)
        ) / 8
        print(f'  threshold={thresh}: avg {n_act:.1f} features active per image')
