"""
Ridge regression neural prediction (Pearson r, 70-170ms, ROI-filtered, NC-normalised).

Conditions:
  Raw B7 / Raw B9 / Raw B11
  B7/B9 SpSAE Full/G0/G1   (ckpts_small_spatial_block7/9,  dict=512, 2 groups)
  B7/B9 StdSAE Full/G0/G1  (ckpts_small_standard_block7/9, dict=512, 2 groups)
  B7/B9 MultSAE Full/G0-G3 (ckpts_block_sweep/block7/9,    dict=512, 4 groups)

Plots:
  1. PSTH time-series (mean Pearson r vs time, NC>=thresh units)
  2. Window-averaged bar chart: early (70-120ms) solid, late (120-170ms) hatched, NC-normalised
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
OUTDIR      = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results/ridge_pred')

CKPT = {
    'SpSAE_B7':   BASE / 'ckpts_small_spatial_block7/ae_final.pt',
    'SpSAE_B9':   BASE / 'ckpts_small_spatial_block9/ae_final.pt',
    'StdSAE_B7':  BASE / 'ckpts_small_standard_block7/ae_final.pt',
    'StdSAE_B9':  BASE / 'ckpts_small_standard_block9/ae_final.pt',
    'MultSAE_B7': BASE / 'ckpts_block_sweep/block7/ae_final.pt',
    'MultSAE_B9': BASE / 'ckpts_block_sweep/block9/ae_final.pt',
}

N_NSD       = 1000
PCA_DIM     = 200
T_START_MS  = 70.0
T_END_MS    = 170.0
T_EARLY_END = 120.0
NC_THRESH   = 0.4
NC_SPLITS   = 10
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
RIDGE_ALPHAS = np.logspace(-3, 6, 20)

sys.path.insert(0, str(BASE / 'temporal-saes/dictionary_learning'))
from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE

# ── ROI / NC helpers ───────────────────────────────────────────────────────

def load_exclude_df():
    df = pd.read_excel(EXCLUDE_XLS, header=None)
    df.columns = ['SesIdx','y1','y2','AREALABEL','RoiIndex','Category','Area']
    df = df[df['SesIdx'].apply(lambda x: str(x).isdigit())]
    df['SesIdx'] = df['SesIdx'].astype(int)
    return df


def load_neural_session(mat_path, ses_idx, exclude_df, t_start_ms, t_end_ms):
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
        resp = np.stack(resp_list, axis=0).astype(np.float32)  # (n_roi, n_time, N_NSD)
    n_roi, n_time, _ = resp.shape
    t_axis = np.linspace(-49, 400, n_time)
    rng = np.random.default_rng(42)
    train_idx = rng.permutation(N_NSD)[:int(N_NSD * 0.8)]
    return dict(responses=resp, train_idx=train_idx, t_axis=t_axis,
                n_units=n_roi, n_units_total=n_units, roi_mask=roi_mask)


def compute_nc(mat_path, t_start_ms, t_end_ms, n_splits=10, seed=42):
    rng = np.random.default_rng(seed)
    with h5py.File(mat_path, 'r') as f:
        md = f['meta_data']
        dvi = md['dataset_valid_idx'][()].flatten()
        tvi = md['trial_valid_idx'][()].flatten()
        valid_mask = dvi.astype(bool)
        valid_img_idx = tvi[valid_mask].astype(int)
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
            time_avg = raster[t0_idx:t1_idx+1, :].mean(axis=0)
            r_splits = []
            for _ in range(n_splits):
                h1, h2 = [], []
                for img_i, trials in img_to_trials.items():
                    if len(trials) < 2: continue
                    t_arr = np.array(trials); rng.shuffle(t_arr)
                    mid = len(t_arr) // 2
                    h1.append(time_avg[t_arr[:mid]].mean())
                    h2.append(time_avg[t_arr[mid:]].mean())
                if len(h1) < 10: r_splits.append(0.0); continue
                h1, h2 = np.array(h1), np.array(h2)
                r = float(max(np.corrcoef(h1, h2)[0,1], 0.0))
                r_splits.append(2*r/(1+r) if (1+r)>0 else 0.0)
            nc[u] = float(np.mean(r_splits))
    return nc


# ── Encoding ───────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_patches(ae, patches, device, batch=32):
    ae = ae.to(device).eval()
    N, P, D = patches.shape
    out = []
    for s in range(0, N, batch):
        x = torch.tensor(patches[s:s+batch].reshape(-1, D), dtype=torch.float32, device=device)
        f = ae.encode(x, use_threshold=False)
        out.append(f.cpu().numpy().reshape(-1, P, ae.dict_size))
    ae.cpu(); return np.concatenate(out, 0)


def load_patches(block):
    path = CACHE_DIR / f'dinov2_spatial_block{block}.pkl'
    with open(path, 'rb') as f:
        d = pickle.load(f)
    key = [k for k in d if 'patch_spatial' in k][0]
    return d[key].astype(np.float32)[:N_NSD]   # (N_NSD, P, D)


# ── Ridge regression ───────────────────────────────────────────────────────

def ridge_pearson_r(X, responses, train_idx, time_idx, pca_dim=200):
    """Returns (n_time, n_units) Pearson r using RidgeCV (multi-output).
    responses: (n_units, n_time_total, N_NSD)
    """
    test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)
    n_comp = min(pca_dim, X.shape[1], len(train_idx) - 1)
    pca = PCA(n_components=n_comp)
    Xtr = pca.fit_transform(X[train_idx])
    Xte = pca.transform(X[test_idx])
    n_t, n_u = len(time_idx), responses.shape[0]
    r_arr = np.full((n_t, n_u), np.nan, np.float32)
    for ti, tidx in enumerate(time_idx):
        Ytr = responses[:, tidx, train_idx].T   # (n_train, n_units)
        Yte = responses[:, tidx, test_idx].T    # (n_test,  n_units)
        # Skip units with no variance in training set
        good = Ytr.std(axis=0) > 1e-6
        if good.sum() == 0:
            continue
        clf = RidgeCV(alphas=RIDGE_ALPHAS, cv=3)
        clf.fit(Xtr, Ytr[:, good])
        Yhat = clf.predict(Xte)   # (n_test, n_good)
        good_idx = np.where(good)[0]
        for k, u in enumerate(good_idx):
            if Yte[:, u].std() > 1e-6 and Yhat[:, k].std() > 1e-6:
                r_arr[ti, u] = float(np.corrcoef(Yte[:, u], Yhat[:, k])[0, 1])
    return r_arr


# ── Plotting ───────────────────────────────────────────────────────────────

COND_COLORS = {
    'Raw B7':          '#444444',
    'Raw B9':          '#777777',
    'Raw B11':         '#000000',
    'B7 SpSAE Full':   '#1565C0',
    'B7 SpSAE G0':     '#1E88E5',
    'B7 SpSAE G1':     '#90CAF9',
    'B9 SpSAE Full':   '#1B5E20',
    'B9 SpSAE G0':     '#43A047',
    'B9 SpSAE G1':     '#A5D6A7',
    'B7 StdSAE Full':  '#E65100',
    'B7 StdSAE G0':    '#FB8C00',
    'B7 StdSAE G1':    '#FFCC80',
    'B9 StdSAE Full':  '#880E4F',
    'B9 StdSAE G0':    '#E91E63',
    'B9 StdSAE G1':    '#F48FB1',
    'B7 MultSAE Full': '#4A148C',
    'B7 MultSAE G0':   '#7B1FA2',
    'B7 MultSAE G1':   '#AB47BC',
    'B7 MultSAE G2':   '#CE93D8',
    'B7 MultSAE G3':   '#E1BEE7',
    'B9 MultSAE Full': '#004D40',
    'B9 MultSAE G0':   '#00897B',
    'B9 MultSAE G1':   '#4DB6AC',
    'B9 MultSAE G2':   '#80CBC4',
    'B9 MultSAE G3':   '#B2DFDB',
}


def plot_psth(all_r, t_axis, nc_thresh, outpath):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axvspan(70, 120, alpha=0.12, color='blue')
    ax.axvspan(120, 170, alpha=0.12, color='green')
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    for name, mat in all_r.items():
        if mat is None or mat.shape[0] == 0: continue
        mean_r = np.nanmean(mat, axis=0)
        sem_r  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0).clip(1))
        c = COND_COLORS.get(name, '#888888')
        ls = '--' if any(g in name for g in ('G1','G2','G3')) else '-'
        ax.plot(t_axis, mean_r, color=c, linewidth=1.2, linestyle=ls, label=name)
        ax.fill_between(t_axis, mean_r - sem_r, mean_r + sem_r, color=c, alpha=0.1)
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Mean Pearson r (NC-normalised)', fontsize=11)
    ax.set_title(f'Ridge Neural Prediction — NC≥{nc_thresh} units', fontsize=12)
    ax.legend(fontsize=6.5, ncol=3, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', outpath)


def plot_window_bars(all_r_early, all_r_late, nc_thresh, outpath):
    conds   = list(all_r_early.keys())
    means_e = [np.nanmean(all_r_early[c]) if all_r_early[c] else 0 for c in conds]
    sems_e  = [np.nanstd(all_r_early[c]) / np.sqrt(max(1, len(all_r_early[c]))) if all_r_early[c] else 0 for c in conds]
    means_l = [np.nanmean(all_r_late[c])  if all_r_late[c]  else 0 for c in conds]
    sems_l  = [np.nanstd(all_r_late[c])  / np.sqrt(max(1, len(all_r_late[c])))  if all_r_late[c]  else 0 for c in conds]
    x      = np.arange(len(conds))
    w      = 0.38
    colors = [COND_COLORS.get(c, '#888888') for c in conds]
    fig, ax = plt.subplots(figsize=(max(len(conds) * 0.95 + 1, 14), 6))
    ax.bar(x - w/2, means_e, w, yerr=sems_e, color=colors, alpha=0.9,
           capsize=3, label='Early (70–120 ms)', error_kw=dict(elinewidth=1))
    ax.bar(x + w/2, means_l, w, yerr=sems_l, color=colors, alpha=0.55,
           capsize=3, label='Late (120–170 ms)', hatch='///', error_kw=dict(elinewidth=1))
    ymax = max(max(means_e), max(means_l)) * 1.45 + 0.01
    for i, (me, ml, se, sl) in enumerate(zip(means_e, means_l, sems_e, sems_l)):
        ax.text(i - w/2, me + se + ymax*0.015, f'{me:.3f}', ha='center', va='bottom',
                fontsize=6.5, rotation=40, rotation_mode='anchor')
        ax.text(i + w/2, ml + sl + ymax*0.015, f'{ml:.3f}', ha='center', va='bottom',
                fontsize=6.5, rotation=40, rotation_mode='anchor')
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('NC-normalised mean Pearson r', fontsize=11)
    ax.set_title(f'Ridge Neural Prediction — Window-averaged (NC≥{nc_thresh})', fontsize=11)
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(pad=1.5)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', outpath)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    exclude_df = load_exclude_df()

    print('Loading DINOv2 patches...')
    raw_patches = {b: load_patches(b) for b in [7, 9, 11]}
    N = N_NSD

    print('Loading and encoding SAE checkpoints...')
    features = {}
    for b in [7, 9, 11]:
        features[f'Raw B{b}'] = raw_patches[b].reshape(N, -1)

    sae_defs = {
        'SpSAE_B7':   (7,  [0, 1]),
        'SpSAE_B9':   (9,  [0, 1]),
        'StdSAE_B7':  (7,  [0, 1]),
        'StdSAE_B9':  (9,  [0, 1]),
        'MultSAE_B7': (7,  [0, 1, 2, 3]),
        'MultSAE_B9': (9,  [0, 1, 2, 3]),
    }
    for sae_name, (block, groups) in sae_defs.items():
        print(f'  Encoding {sae_name}...')
        ae = TemporalMatryoshkaBatchTopKSAE.from_pretrained(str(CKPT[sae_name]), temporal=True)
        codes = encode_patches(ae, raw_patches[block], DEVICE)  # (N, P, dict_size)
        gs = ae.group_sizes.tolist()
        g_bounds = [0] + list(np.cumsum(gs))
        prefix = sae_name.replace('SAE_', 'SAE ').replace('_', ' ')
        features[f'{prefix} Full'] = codes.reshape(N, -1)
        for gi in groups:
            features[f'{prefix} G{gi}'] = codes[:, :, g_bounds[gi]:g_bounds[gi+1]].reshape(N, -1)
        del ae, codes

    cond_names = list(features.keys())
    print(f'Conditions: {cond_names}')

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
            nc_mask = nc >= NC_THRESH
            t_axis_ses = np.load(ses_outdir / 't_axis.npy')
            t_mask_ses = (t_axis_ses >= T_START_MS) & (t_axis_ses <= T_END_MS)
            for c in cond_names:
                fp = ses_outdir / f'r_{c.replace(" ","_")}.npy'
                if fp.exists():
                    r_tb = np.load(fp)
                    r_nc = r_tb[:, nc_mask] / nc[nc_mask][None, :]
                    for u in range(r_nc.shape[1]):
                        all_r_psth[c].append(r_nc[:, u])
                    e_mask = (t_axis_ses[t_mask_ses] >= 70) & (t_axis_ses[t_mask_ses] <= 120)
                    l_mask = (t_axis_ses[t_mask_ses] >= 120) & (t_axis_ses[t_mask_ses] <= 170)
                    early_vals = np.nanmean(r_nc[e_mask, :], axis=0)
                    late_vals  = np.nanmean(r_nc[l_mask, :], axis=0)
                    all_r_early[c].extend(early_vals[~np.isnan(early_vals)].tolist())
                    all_r_late[c].extend(late_vals[~np.isnan(late_vals)].tolist())
            print(f'[{ses_idx}] SKIP (cached): {session}')
            continue

        print(f'\n{"="*60}')
        print(f'[{ses_idx}/{len(mat_files)}] {session}')
        t0 = time.time()

        neural = load_neural_session(mat_path, ses_idx, exclude_df, T_START_MS, T_END_MS)
        if neural is None:
            print('  SKIP: no ROI units.')
            continue

        responses  = neural['responses']
        train_idx  = neural['train_idx']
        t_axis     = neural['t_axis']
        t_mask     = (t_axis >= T_START_MS) & (t_axis <= T_END_MS)
        time_idx   = np.where(t_mask)[0]
        t_axis_win = t_axis[t_mask]
        print(f'  n_roi={neural["n_units"]}/{neural["n_units_total"]}  t_bins={len(time_idx)}')

        print('  Computing NC...')
        nc = compute_nc(mat_path, T_START_MS, T_END_MS, NC_SPLITS)
        roi_mask = neural['roi_mask']
        nc_roi   = nc[roi_mask]
        nc_mask  = nc_roi >= NC_THRESH
        print(f'  NC>=0.4: {nc_mask.sum()}/{len(nc_roi)}')
        np.save(ses_outdir / 'nc.npy', nc_roi)
        np.save(ses_outdir / 't_axis.npy', t_axis_win)

        for c in cond_names:
            X = features[c]
            print(f'  [{c}] Ridge...')
            r_tb = ridge_pearson_r(X, responses, train_idx, time_idx, PCA_DIM)
            np.save(ses_outdir / f'r_{c.replace(" ","_")}.npy', r_tb)
            r_nc = r_tb[:, nc_mask] / nc_roi[nc_mask][None, :]
            e_mask = (t_axis_win >= 70)  & (t_axis_win <= 120)
            l_mask = (t_axis_win >= 120) & (t_axis_win <= 170)
            for u in range(r_nc.shape[1]):
                all_r_psth[c].append(r_nc[:, u])
            early_vals = np.nanmean(r_nc[e_mask, :], axis=0)
            late_vals  = np.nanmean(r_nc[l_mask, :], axis=0)
            all_r_early[c].extend(early_vals[~np.isnan(early_vals)].tolist())
            all_r_late[c].extend(late_vals[~np.isnan(late_vals)].tolist())

        elapsed = (time.time() - t0) / 60
        print(f'  Done in {elapsed:.1f} min')
        done_flag.touch()

    # Plots
    t_axis_plot = np.linspace(T_START_MS, T_END_MS,
                              list(all_r_psth.values())[0][0].shape[0])
    all_r_mean = {c: np.stack(v, axis=0) for c, v in all_r_psth.items() if v}

    plot_psth(all_r_mean, t_axis_plot, NC_THRESH, OUTDIR / 'psth_timeseries.png')
    plot_window_bars(all_r_early, all_r_late, NC_THRESH, OUTDIR / 'window_bars.png')

    print('\nAll done.')


if __name__ == '__main__':
    main()
