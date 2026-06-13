"""
Ridge regression neural prediction — NO mean-pool, spatial flatten.

Uses ckpts_multiscale_v1 (dict=16384, 4 groups [4096 each], group_steps=[4,2,1,0])
trained on DINOv2 final x_norm_patchtokens.

Feature vectors per image (no mean-pool):
  Raw final:  256 patches × 768  = 196 608 dims
  Gk:         256 patches × 4096 = 1 048 576 dims  (randomized PCA → 200 PCs)

Plots same as ridge_pred: PSTH timeseries + window-averaged bars.
"""

import os, sys, pickle, time, h5py
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path('/n/home12/binxuwang/T-SAE-Spatial')
NHD_DIR     = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
DATA_DIR    = NHD_DIR / 'Datasets/NSD_N3'
CACHE_DIR   = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/cache')
EXCLUDE_XLS = NHD_DIR / 'Datasets/Triple_N/Others/exclude_area.xls'
OUTDIR      = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results/ridge_spatial')
DINO_REPO   = str(NHD_DIR / 'torch_cache/hub/facebookresearch_dinov2_main')

CKPT_MULTISCALE = BASE / 'ckpts_multiscale_v1/ae_final.pt'

N_NSD       = 1000
PCA_DIM     = 200
T_START_MS  = 70.0
T_END_MS    = 170.0
NC_THRESH   = 0.4
NC_SPLITS   = 10
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
RIDGE_ALPHAS = np.logspace(-3, 6, 20)

sys.path.insert(0, str(BASE / 'temporal-saes/dictionary_learning'))
from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE

# ── DINOv2 final norm patch token extraction ───────────────────────────────

FINAL_NORM_CACHE = CACHE_DIR / 'dinov2_final_norm_patchtokens.pkl'
NSD_IMG_DIR = NHD_DIR / 'Datasets/NSD_N3'

def _get_nsd_image_paths():
    """Return sorted list of NSD image paths (COCO 2017)."""
    from torchvision import transforms
    # Load from the .mat files to get the same 1000 images used in prediction
    # Actually, patches are precomputed per NSD image index.
    # The cache stores (1072, 256, 768) but we need only the first N_NSD=1000.
    return None   # handled below via existing block cache + DINO re-extraction

def build_final_norm_cache(n_images=N_NSD):
    """Extract x_norm_patchtokens from DINOv2 for NSD images, cache to disk."""
    import torch
    from torchvision import transforms
    from PIL import Image

    print(f'Building final norm patch token cache for {n_images} images...')

    # Load existing block-11 cache to get image ordering (same images)
    b11_cache = pickle.load(open(CACHE_DIR / 'dinov2_spatial_block11.pkl', 'rb'))
    img_paths = b11_cache.get('image_paths', None)

    # Load DINOv2 (with registers for consistency with block_sweep training)
    model = torch.hub.load(DINO_REPO, 'dinov2_vitb14_reg', source='local', trust_repo=True)
    model = model.to(DEVICE).eval()

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    all_patches = []
    with torch.no_grad():
        for i in range(n_images):
            if img_paths is not None:
                pil = Image.open(img_paths[i]).convert('RGB')
                tensor = preprocess(pil).unsqueeze(0).to(DEVICE)
            else:
                raise RuntimeError('No image paths in cache — cannot rebuild final norm cache.')
            out = model.forward_features(tensor)
            patches = out['x_norm_patchtokens'].squeeze(0).cpu().numpy()  # (256, 768)
            all_patches.append(patches)
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{n_images}')

    arr = np.stack(all_patches, axis=0).astype(np.float32)  # (N, 256, 768)
    del model
    torch.cuda.empty_cache()

    with open(FINAL_NORM_CACHE, 'wb') as f:
        pickle.dump({'x_norm_patchtokens': arr}, f, protocol=4)
    print(f'Saved: {FINAL_NORM_CACHE}  shape={arr.shape}')
    return arr


def load_final_norm_patches():
    if FINAL_NORM_CACHE.exists():
        d = pickle.load(open(FINAL_NORM_CACHE, 'rb'))
        arr = d['x_norm_patchtokens']
        print(f'Loaded final norm cache: {arr.shape}')
        return arr.astype(np.float32)[:N_NSD]
    else:
        return build_final_norm_cache(N_NSD)


# ── SAE encoding ───────────────────────────────────────────────────────────

@torch.no_grad()
def encode_patches_spatial(ae, patches, device, batch=16):
    """Encode (N, P, D) patches → (N, P, dict_size), preserving spatial dim."""
    ae = ae.to(device).eval()
    N, P, D = patches.shape
    out = []
    for s in range(0, N, batch):
        x = torch.tensor(patches[s:s+batch].reshape(-1, D), dtype=torch.float32, device=device)
        f = ae.encode(x, use_threshold=False)   # (batch*P, dict_size)
        out.append(f.cpu().numpy().reshape(-1, P, ae.dict_size))
    ae.cpu()
    return np.concatenate(out, 0)   # (N, P, dict_size)


# ── ROI / NC (same as ridge_pred.py) ──────────────────────────────────────

def load_exclude_df():
    df = pd.read_excel(EXCLUDE_XLS, header=None)
    df.columns = ['SesIdx','y1','y2','AREALABEL','RoiIndex','Category','Area']
    df = df[df['SesIdx'].apply(lambda x: str(x).isdigit())]
    df['SesIdx'] = df['SesIdx'].astype(int)
    return df


def load_neural_session(mat_path, ses_idx, exclude_df):
    ses_rows = exclude_df[exclude_df['SesIdx'] == ses_idx]
    with h5py.File(mat_path, 'r') as f:
        gu = f['GoodUnitStrc']
        n_units = gu['Raster'].shape[0]
        sp_y = np.array([f[gu['spikepos'][i,0]][()].flatten()[1] for i in range(n_units)])
        if len(ses_rows) == 0:
            roi_mask = np.ones(n_units, dtype=bool)
        else:
            roi_mask = np.zeros(n_units, dtype=bool)
            for _, row in ses_rows.iterrows():
                roi_mask |= (sp_y >= float(row['y1'])) & (sp_y <= float(row['y2']))
        unit_idx = np.where(roi_mask)[0]
        resp_refs = gu['response_matrix_img']
        resp_list = [f[resp_refs[ui, 0]][()][:, :N_NSD] for ui in unit_idx]
        if not resp_list:
            return None
        resp = np.stack(resp_list, axis=0).astype(np.float32)
    t_axis = np.linspace(-49, 400, resp.shape[1])
    rng = np.random.default_rng(42)
    train_idx = rng.permutation(N_NSD)[:int(N_NSD * 0.8)]
    return dict(responses=resp, train_idx=train_idx, t_axis=t_axis,
                n_units=resp.shape[0], n_units_total=n_units, roi_mask=roi_mask)


def compute_nc(mat_path, t_start_ms, t_end_ms, n_splits=10, seed=42):
    rng = np.random.default_rng(seed)
    with h5py.File(mat_path, 'r') as f:
        md = f['meta_data']
        valid_img_idx = f['meta_data']['trial_valid_idx'][()].flatten()[
            f['meta_data']['dataset_valid_idx'][()].flatten().astype(bool)].astype(int)
        img_to_trials = {}
        for ti, img_i in enumerate(valid_img_idx):
            if img_i < 1: continue
            img_to_trials.setdefault(img_i, []).append(ti)
        gu = f['GoodUnitStrc']
        n_units = gu['Raster'].shape[0]
        t_axis = np.linspace(-49, 400, f[gu['Raster'][0,0]][()].shape[0])
        t_mask = (t_axis >= t_start_ms) & (t_axis <= t_end_ms)
        t0_idx, t1_idx = np.where(t_mask)[0][[0,-1]]
        nc = np.zeros(n_units, dtype=np.float32)
        for u in range(n_units):
            raster = f[gu['Raster'][u,0]][()]
            time_avg = raster[t0_idx:t1_idx+1].mean(axis=0)
            r_splits = []
            for _ in range(n_splits):
                h1, h2 = [], []
                for img_i, trials in img_to_trials.items():
                    if len(trials) < 2: continue
                    t_arr = np.array(trials); rng.shuffle(t_arr)
                    mid = len(t_arr)//2
                    h1.append(time_avg[t_arr[:mid]].mean())
                    h2.append(time_avg[t_arr[mid:]].mean())
                if len(h1) < 10: r_splits.append(0.0); continue
                h1, h2 = np.array(h1), np.array(h2)
                r = float(max(np.corrcoef(h1, h2)[0,1], 0.0))
                r_splits.append(2*r/(1+r) if (1+r)>0 else 0.0)
            nc[u] = float(np.mean(r_splits))
    return nc


# ── Ridge regression (no mean-pool, spatial flatten) ──────────────────────

def ridge_pearson_r_spatial(X, responses, train_idx, time_idx, pca_dim=200):
    """
    X: (N_NSD, n_feats)  — flattened spatial features, no mean-pool
    responses: (n_units, n_time_total, N_NSD)
    Returns (n_time, n_units) Pearson r.
    """
    test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)
    n_feats = X.shape[1]
    n_comp = min(pca_dim, n_feats, len(train_idx) - 1)
    # Use randomized SVD for very high-dim features
    solver = 'randomized' if n_feats > 5000 else 'full'
    pca = PCA(n_components=n_comp, svd_solver=solver, random_state=0)
    Xtr = pca.fit_transform(X[train_idx])
    Xte = pca.transform(X[test_idx])

    n_t, n_u = len(time_idx), responses.shape[0]
    r_arr = np.full((n_t, n_u), np.nan, np.float32)
    for ti, tidx in enumerate(time_idx):
        Ytr = responses[:, tidx, train_idx].T   # (n_train, n_units)
        Yte = responses[:, tidx, test_idx].T
        good = Ytr.std(axis=0) > 1e-6
        if good.sum() == 0:
            continue
        clf = RidgeCV(alphas=RIDGE_ALPHAS, cv=3)
        clf.fit(Xtr, Ytr[:, good])
        Yhat = clf.predict(Xte)
        good_idx = np.where(good)[0]
        for k, u in enumerate(good_idx):
            if Yte[:, u].std() > 1e-6 and Yhat[:, k].std() > 1e-6:
                r_arr[ti, u] = float(np.corrcoef(Yte[:, u], Yhat[:, k])[0, 1])
    return r_arr


# ── Plotting ───────────────────────────────────────────────────────────────

COND_COLORS = {
    'Raw final':    '#000000',
    'MultSAE G0':   '#7B1FA2',
    'MultSAE G1':   '#AB47BC',
    'MultSAE G2':   '#CE93D8',
    'MultSAE G3':   '#E1BEE7',
    'MultSAE Full': '#4A148C',
}


def plot_psth(all_r, t_axis, nc_thresh, outpath):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvspan(70, 120, alpha=0.12, color='blue')
    ax.axvspan(120, 170, alpha=0.12, color='green')
    for name, mat in all_r.items():
        if mat is None or mat.shape[0] == 0: continue
        mean_r = np.nanmean(mat, axis=0)
        sem_r  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0).clip(1))
        c = COND_COLORS.get(name, '#888888')
        ls = '--' if any(g in name for g in ('G1','G2','G3')) else '-'
        ax.plot(t_axis, mean_r, color=c, linewidth=1.5, linestyle=ls, label=name)
        ax.fill_between(t_axis, mean_r-sem_r, mean_r+sem_r, color=c, alpha=0.12)
    ax.set_xlabel('Time (ms)', fontsize=11); ax.set_ylabel('NC-norm Pearson r', fontsize=11)
    ax.set_title(f'Ridge Prediction (no mean-pool, MultSAE dict=16384) — NC≥{nc_thresh}', fontsize=11)
    ax.legend(fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('Saved:', outpath)


def plot_window_bars(all_r_early, all_r_late, nc_thresh, outpath):
    conds   = list(COND_COLORS.keys())
    conds   = [c for c in conds if c in all_r_early and all_r_early[c]]
    means_e = [np.nanmean(all_r_early[c]) for c in conds]
    sems_e  = [np.nanstd(all_r_early[c])/np.sqrt(max(1,len(all_r_early[c]))) for c in conds]
    means_l = [np.nanmean(all_r_late[c])  for c in conds]
    sems_l  = [np.nanstd(all_r_late[c]) /np.sqrt(max(1,len(all_r_late[c])))  for c in conds]
    x = np.arange(len(conds)); w = 0.38
    colors = [COND_COLORS[c] for c in conds]
    fig, ax = plt.subplots(figsize=(max(len(conds)*1.5+1, 10), 6))
    ax.bar(x-w/2, means_e, w, yerr=sems_e, color=colors, alpha=0.9, capsize=3,
           label='Early (70–120ms)', error_kw=dict(elinewidth=1))
    ax.bar(x+w/2, means_l, w, yerr=sems_l, color=colors, alpha=0.55, capsize=3,
           label='Late (120–170ms)', hatch='///', error_kw=dict(elinewidth=1))
    ymax = max(max(means_e), max(means_l)) * 1.45 + 0.01
    for i,(me,ml,se,sl) in enumerate(zip(means_e,means_l,sems_e,sems_l)):
        ax.text(i-w/2, me+se+ymax*0.015, f'{me:.3f}', ha='center', va='bottom', fontsize=9, rotation=40, rotation_mode='anchor')
        ax.text(i+w/2, ml+sl+ymax*0.015, f'{ml:.3f}', ha='center', va='bottom', fontsize=9, rotation=40, rotation_mode='anchor')
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('NC-norm mean Pearson r', fontsize=11)
    ax.set_title(f'Ridge Prediction no mean-pool — Window-averaged (NC≥{nc_thresh})', fontsize=11)
    ax.set_ylim(0, ymax); ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=1.5); fig.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('Saved:', outpath)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    exclude_df = load_exclude_df()

    print('Loading final norm patch tokens...')
    raw_patches = load_final_norm_patches()   # (N_NSD, 256, 768)
    N, P, D = raw_patches.shape

    print('Loading and encoding MultSAE (dict=16384)...')
    ae = TemporalMatryoshkaBatchTopKSAE.from_pretrained(str(CKPT_MULTISCALE), temporal=True)
    codes = encode_patches_spatial(ae, raw_patches, DEVICE)  # (N, P, 16384)
    gs = ae.group_sizes.tolist()
    g_bounds = [0] + list(np.cumsum(gs))
    print(f'  group_sizes: {gs}  g_bounds: {g_bounds}')

    # Build feature dict — flatten spatial dim: (N, P*feat_dim)
    features = {}
    features['Raw final']    = raw_patches.reshape(N, -1)            # (N, 196608)
    features['MultSAE Full'] = codes.reshape(N, -1)                   # (N, 4194304) — skip if OOM
    for gi in range(len(gs)):
        feat_g = codes[:, :, g_bounds[gi]:g_bounds[gi+1]].reshape(N, -1)
        features[f'MultSAE G{gi}'] = feat_g

    # Drop Full if too large for available RAM (>8GB)
    full_size_gb = codes.reshape(N, -1).nbytes / 1e9
    if full_size_gb > 8.0:
        print(f'  MultSAE Full is {full_size_gb:.1f} GB — skipping to save memory')
        del features['MultSAE Full']
    del ae, codes

    cond_names = list(features.keys())
    print(f'Conditions: {cond_names}')
    for c, v in features.items():
        print(f'  {c}: {v.shape}  ({v.nbytes/1e9:.2f} GB)')

    all_r_psth  = {c: [] for c in cond_names}
    all_r_early = {c: [] for c in cond_names}
    all_r_late  = {c: [] for c in cond_names}

    mat_files = sorted(DATA_DIR.glob('GoodUnit_*.mat'))
    print(f'\n{len(mat_files)} sessions found.')

    for idx, mat_path in enumerate(mat_files):
        ses_idx = idx + 1
        session = mat_path.stem
        ses_outdir = OUTDIR / session
        ses_outdir.mkdir(exist_ok=True)
        done_flag = ses_outdir / 'DONE.flag'

        if done_flag.exists():
            nc = np.load(ses_outdir / 'nc.npy')
            t_axis = np.load(ses_outdir / 't_axis.npy')
            nc_mask = nc >= NC_THRESH
            e_mask = (t_axis >= 70) & (t_axis <= 120)
            l_mask = (t_axis >= 120) & (t_axis <= 170)
            for c in cond_names:
                fp = ses_outdir / f'r_{c.replace(" ","_")}.npy'
                if fp.exists():
                    r_tb = np.load(fp)
                    r_nc = r_tb[:, nc_mask] / nc[nc_mask][None, :]
                    for u in range(r_nc.shape[1]): all_r_psth[c].append(r_nc[:, u])
                    all_r_early[c].extend(np.nanmean(r_nc[e_mask], axis=0).tolist())
                    all_r_late[c].extend(np.nanmean(r_nc[l_mask], axis=0).tolist())
            print(f'[{ses_idx}] SKIP (cached): {session}')
            continue

        print(f'\n{"="*60}\n[{ses_idx}/{len(mat_files)}] {session}')
        t0 = time.time()

        neural = load_neural_session(mat_path, ses_idx, exclude_df)
        if neural is None:
            print('  SKIP: no ROI units.'); continue

        responses = neural['responses']
        train_idx = neural['train_idx']
        t_axis    = neural['t_axis']
        t_mask    = (t_axis >= T_START_MS) & (t_axis <= T_END_MS)
        time_idx  = np.where(t_mask)[0]
        t_win     = t_axis[t_mask]
        print(f'  n_roi={neural["n_units"]}/{neural["n_units_total"]}  t_bins={len(time_idx)}')

        print('  Computing NC...')
        nc = compute_nc(mat_path, T_START_MS, T_END_MS, NC_SPLITS)
        roi_mask = neural['roi_mask']
        nc_roi   = nc[roi_mask]
        nc_mask  = nc_roi >= NC_THRESH
        print(f'  NC>=0.4: {nc_mask.sum()}/{len(nc_roi)}')
        np.save(ses_outdir / 'nc.npy', nc_roi)
        np.save(ses_outdir / 't_axis.npy', t_win)

        e_mask = (t_win >= 70) & (t_win <= 120)
        l_mask = (t_win >= 120) & (t_win <= 170)

        for c in cond_names:
            print(f'  [{c}] Ridge (no pool, {features[c].shape[1]} feats → PCA {PCA_DIM})...')
            r_tb = ridge_pearson_r_spatial(features[c], responses, train_idx, time_idx, PCA_DIM)
            np.save(ses_outdir / f'r_{c.replace(" ","_")}.npy', r_tb)
            r_nc = r_tb[:, nc_mask] / nc_roi[nc_mask][None, :]
            for u in range(r_nc.shape[1]): all_r_psth[c].append(r_nc[:, u])
            all_r_early[c].extend(np.nanmean(r_nc[e_mask], axis=0).tolist())
            all_r_late[c].extend(np.nanmean(r_nc[l_mask], axis=0).tolist())

        print(f'  Done in {(time.time()-t0)/60:.1f} min')
        done_flag.touch()

    t_axis_plot = np.load(list((OUTDIR).glob('*/t_axis.npy'))[0])
    all_r_mean = {c: np.stack(v, axis=0) for c, v in all_r_psth.items() if v}
    plot_psth(all_r_mean, t_axis_plot, NC_THRESH, OUTDIR / 'psth_timeseries.png')
    plot_window_bars(all_r_early, all_r_late, NC_THRESH, OUTDIR / 'window_bars.png')
    print('\nAll done.')


if __name__ == '__main__':
    main()
