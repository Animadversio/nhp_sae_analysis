"""
Ridge regression neural prediction for Independent Sub-SAE checkpoint.
dict_size=16384, 4 groups (G0-G3, 4096 each), k=32, group_steps=[3,2,1,0], final norm tokens.
Conditions: Full, G0, G1, G2, G3 (mean-pooled across 256 patches).
"""

import os, sys, pickle, time, h5py
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import pandas as pd

BASE     = Path('/n/home12/binxuwang/T-SAE-Spatial')
NHD_DIR  = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
DATA_DIR = NHD_DIR / 'Datasets/NSD_N3'
CACHE_DIR = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/cache')
EXCLUDE_XLS = NHD_DIR / 'Datasets/Triple_N/Others/exclude_area.xls'
OUTDIR   = Path('/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Projects/nhp_sae_analysis/results/ridge_independent_subsae')
OUTDIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH  = BASE / 'ckpts_independent_subsae_v1/ae_final.pt'
DICT_SIZE  = 16384
N_GROUPS   = 4
GROUP_SIZE = DICT_SIZE // N_GROUPS  # 4096
K          = 32

N_NSD       = 1000
PCA_DIM     = 200
T_START_MS  = 70.0
T_END_MS    = 170.0
NC_THRESH   = 0.4
NC_SPLITS   = 10
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
RIDGE_ALPHAS = np.logspace(-3, 6, 20)

COND_NAMES = ['Full', 'G0', 'G1', 'G2', 'G3']
COND_LABELS = {
    'Full': 'Full SAE',
    'G0':   'G0 (dist=3)',
    'G1':   'G1 (dist=2)',
    'G2':   'G2 (dist=1)',
    'G3':   'G3 (recon-only)',
}
COND_COLORS = {
    'Full': '#1A237E',
    'G0':   '#1565C0',
    'G1':   '#1976D2',
    'G2':   '#42A5F5',
    'G3':   '#90CAF9',
}

sys.path.insert(0, str(BASE / 'temporal-saes/dictionary_learning'))
from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE

# ── Helpers ────────────────────────────────────────────────────────────────────
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
        resp = np.stack(resp_list, axis=0).astype(np.float32)
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

def ridge_pearson_r(X, responses, train_idx, time_idx, pca_dim=200):
    test_idx = np.setdiff1d(np.arange(X.shape[0]), train_idx)
    n_comp = min(pca_dim, X.shape[1], len(train_idx)-1)
    pca = PCA(n_components=n_comp)
    Xtr = pca.fit_transform(X[train_idx])
    Xte = pca.transform(X[test_idx])
    n_t, n_u = len(time_idx), responses.shape[0]
    r_arr = np.full((n_t, n_u), np.nan, np.float32)
    for ti, tidx in enumerate(time_idx):
        Ytr = responses[:, tidx, train_idx].T
        Yte = responses[:, tidx, test_idx].T
        good = Ytr.std(axis=0) > 1e-6
        if good.sum() == 0: continue
        clf = RidgeCV(alphas=RIDGE_ALPHAS, cv=3)
        clf.fit(Xtr, Ytr[:, good])
        Yhat = clf.predict(Xte)
        for k, u in enumerate(np.where(good)[0]):
            if Yte[:, u].std() > 1e-6 and Yhat[:, k].std() > 1e-6:
                r_arr[ti, u] = float(np.corrcoef(Yte[:, u], Yhat[:, k])[0, 1])
    return r_arr

def plot_psth(all_r, t_axis, nc_thresh, outpath):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axvspan(70, 120, alpha=0.10, color='royalblue')
    ax.axvspan(120, 170, alpha=0.10, color='seagreen')
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
    for name in COND_NAMES:
        if not all_r[name]: continue
        mat = np.stack(all_r[name], axis=0)
        mean_r = np.nanmean(mat, axis=0)
        sem_r  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0).clip(1))
        c = COND_COLORS[name]
        lw = 2.2 if name == 'Full' else 1.4
        ls = '--' if name in ('G1','G2','G3') else '-'
        ax.plot(t_axis, mean_r, color=c, lw=lw, ls=ls, label=COND_LABELS[name])
        ax.fill_between(t_axis, mean_r-sem_r, mean_r+sem_r, color=c, alpha=0.12)
        w = (t_axis >= 70) & (t_axis <= 170)
        pk_v = mean_r[w].max()
        pk_t = t_axis[w][np.argmax(mean_r[w])]
        ax.annotate(f'{pk_v:.3f}', xy=(pk_t, pk_v), xytext=(pk_t+1, pk_v+0.005),
                    fontsize=7.5, color=c, fontweight='bold')
    ax.set_xlabel('Time (ms)', fontsize=12); ax.set_ylabel('Mean Pearson r / NC', fontsize=12)
    ax.set_title(f"Ridge Prediction — Independent SubSAE (dict=16384, 4 groups)\nNC≥{nc_thresh}", fontsize=12)
    ax.legend(fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('Saved:', outpath)

def plot_window_bars(all_r_early, all_r_late, nc_thresh, outpath):
    means_e = [np.nanmean(all_r_early[c]) if all_r_early[c] else 0 for c in COND_NAMES]
    sems_e  = [np.nanstd(all_r_early[c])/np.sqrt(max(1,len(all_r_early[c]))) for c in COND_NAMES]
    means_l = [np.nanmean(all_r_late[c]) if all_r_late[c] else 0 for c in COND_NAMES]
    sems_l  = [np.nanstd(all_r_late[c])/np.sqrt(max(1,len(all_r_late[c]))) for c in COND_NAMES]
    x = np.arange(len(COND_NAMES)); w = 0.38
    colors = [COND_COLORS[c] for c in COND_NAMES]
    ymax = max(max(means_e), max(means_l)) * 1.55 + 0.01
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x-w/2, means_e, w, yerr=sems_e, color=colors, alpha=0.92, capsize=4,
           label='Early (70–120 ms)', error_kw=dict(elinewidth=1.2))
    ax.bar(x+w/2, means_l, w, yerr=sems_l, color=colors, alpha=0.55, capsize=4,
           label='Late (120–170 ms)', hatch='///', error_kw=dict(elinewidth=1.2))
    for i,(me,ml,se,sl) in enumerate(zip(means_e,means_l,sems_e,sems_l)):
        ax.text(i-w/2, me+se+ymax*0.015, f'{me:.4f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=colors[i])
        ax.text(i+w/2, ml+sl+ymax*0.015, f'{ml:.4f}', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=colors[i])
    ax.set_xticks(x); ax.set_xticklabels([COND_LABELS[c] for c in COND_NAMES], fontsize=10)
    ax.set_ylabel('Mean Pearson r / NC', fontsize=12); ax.set_ylim(0, ymax)
    ax.set_title(f"Independent SubSAE — Early/Late Window Prediction", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('Saved:', outpath)

def plot_session1(r_ses, t_ax_win, nc_roi, nc_thresh, outpath):
    nc_mask = nc_roi >= nc_thresh
    e_m = (t_ax_win >= 70) & (t_ax_win <= 120)
    l_m = (t_ax_win >= 120) & (t_ax_win <= 170)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in COND_NAMES:
        if name not in r_ses: continue
        r = r_ses[name]  # (n_time, n_roi)
        r_nc = r[:, nc_mask] / nc_roi[nc_mask][None, :]
        mean_r = np.nanmean(r_nc, axis=1)
        c = COND_COLORS[name]
        lw = 2.2 if name == 'Full' else 1.4
        ls = '--' if name in ('G1','G2','G3') else '-'
        axes[0].plot(t_ax_win, mean_r, color=c, lw=lw, ls=ls, label=COND_LABELS[name])
        early_v = float(np.nanmean(r_nc[e_m, :]))
        late_v  = float(np.nanmean(r_nc[l_m, :]))
        axes[1].bar([COND_NAMES.index(name)-0.19], [early_v], 0.38, color=c, alpha=0.92,
                    label=f'{COND_LABELS[name]} early')
        axes[1].bar([COND_NAMES.index(name)+0.19], [late_v], 0.38, color=c, alpha=0.55, hatch='///')
        axes[1].text(COND_NAMES.index(name)-0.19, early_v+0.005, f'{early_v:.4f}',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=c)
        axes[1].text(COND_NAMES.index(name)+0.19, late_v+0.005, f'{late_v:.4f}',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=c)
    axes[0].axvspan(70, 120, alpha=0.10, color='royalblue')
    axes[0].axvspan(120, 170, alpha=0.10, color='seagreen')
    axes[0].axvline(0, color='k', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel('Time (ms)', fontsize=12); axes[0].set_ylabel('Mean Pearson r / NC', fontsize=12)
    axes[0].set_title(f'Session 1 PSTH (NC≥{nc_thresh}, n={nc_mask.sum()} units)', fontsize=11)
    axes[0].legend(fontsize=8); axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
    axes[1].set_xticks(range(len(COND_NAMES))); axes[1].set_xticklabels([COND_LABELS[c] for c in COND_NAMES], fontsize=9, rotation=15)
    axes[1].set_ylabel('Mean Pearson r / NC', fontsize=12)
    axes[1].set_title('Session 1 Early/Late Windows', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3); axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    fig.suptitle('Independent SubSAE — Session 1 Prediction', fontsize=13, fontweight='bold')
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('Saved:', outpath)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load SAE
    print(f"Loading SAE from {CKPT_PATH}...")
    state = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    ae = TemporalMatryoshkaBatchTopKSAE(
        activation_dim=768, dict_size=DICT_SIZE,
        k=K, group_sizes=[GROUP_SIZE]*N_GROUPS, temporal=True,
    )
    ae.load_state_dict(state); ae.eval()
    print(f"  dict_size={DICT_SIZE}, groups={N_GROUPS}×{GROUP_SIZE}, k={K}")

    # Build features (mean-pool over 256 patches)
    FEAT_CACHE = CACHE_DIR / 'indep_subsae_v1_features_meanpool.pkl'
    if FEAT_CACHE.exists():
        print("Loading features from cache...")
        with open(FEAT_CACHE, 'rb') as fh:
            features = pickle.load(fh)
    else:
        DINO_CACHE = CACHE_DIR / 'dinov2_final_norm_patchtokens.pkl'
        print(f"Loading DINOv2 final-norm patch tokens...")
        with open(DINO_CACHE, 'rb') as fh:
            dino_cache = pickle.load(fh)
        patches = dino_cache['x_norm_patchtokens'][:N_NSD]  # (N, 256, 768)
        print(f"  Encoding {N_NSD} images through SAE...")
        ae_gpu = ae.to(DEVICE)
        BATCH = 16
        codes_all = []
        for b in range(0, N_NSD, BATCH):
            x = torch.from_numpy(patches[b:b+BATCH].reshape(-1, 768)).float().to(DEVICE)
            with torch.no_grad():
                c = ae_gpu.encode(x, use_threshold=False).cpu().numpy()
            codes_all.append(c.reshape(-1, 256, DICT_SIZE))
            if (b // BATCH + 1) % 10 == 0:
                print(f"  {b+BATCH}/{N_NSD}")
        ae_gpu.cpu()
        codes = np.concatenate(codes_all, 0)  # (N, 256, DICT_SIZE)
        features = {'Full': codes.mean(1)}
        for g in range(N_GROUPS):
            s, e = g*GROUP_SIZE, (g+1)*GROUP_SIZE
            features[f'G{g}'] = codes[:, :, s:e].mean(1)
        with open(FEAT_CACHE, 'wb') as fh:
            pickle.dump(features, fh)
        print(f"  Saved feature cache: {FEAT_CACHE}")
        print(f"  Feature dims: { {k: v.shape[1] for k,v in features.items()} }")

    exclude_df = load_exclude_df()
    mat_files  = sorted(DATA_DIR.glob('GoodUnit_*.mat'))
    print(f"\n{len(mat_files)} sessions found.\n")

    all_r_psth  = {c: [] for c in COND_NAMES}
    all_r_early = {c: [] for c in COND_NAMES}
    all_r_late  = {c: [] for c in COND_NAMES}
    session1_saved = False

    for si, mat_path in enumerate(mat_files):
        session = mat_path.stem
        ses_dir = OUTDIR / session
        ses_dir.mkdir(exist_ok=True)
        done_flag = ses_dir / 'DONE.flag'

        if done_flag.exists():
            nc = np.load(ses_dir / 'nc.npy')
            nc_mask = nc >= NC_THRESH
            t_ax = np.load(ses_dir / 't_axis.npy')
            e_m = (t_ax >= 70) & (t_ax <= 120)
            l_m = (t_ax >= 120) & (t_ax <= 170)
            for c in COND_NAMES:
                fp = ses_dir / f'r_{c}.npy'
                if fp.exists():
                    r = np.load(fp)
                    r_nc = r[:, nc_mask] / nc[nc_mask][None, :]
                    for u in range(r_nc.shape[1]):
                        all_r_psth[c].append(r_nc[:, u])
                    all_r_early[c].extend(np.nanmean(r_nc[e_m, :], 0).tolist())
                    all_r_late[c].extend(np.nanmean(r_nc[l_m, :], 0).tolist())
            print(f'[{si+1}/{len(mat_files)}] SKIP (cached): {session}')
            continue

        print(f'\n{"="*60}')
        print(f'[{si+1}/{len(mat_files)}] {session}')
        t0 = time.time()

        neural = load_neural_session(mat_path, si+1, exclude_df, T_START_MS, T_END_MS)
        if neural is None:
            print('  SKIP: no ROI units.'); continue

        responses = neural['responses']
        train_idx = neural['train_idx']
        t_axis    = neural['t_axis']
        t_mask    = (t_axis >= T_START_MS) & (t_axis <= T_END_MS)
        time_idx  = np.where(t_mask)[0]
        t_ax_win  = t_axis[t_mask]
        print(f'  n_roi={neural["n_units"]}/{neural["n_units_total"]}  t_bins={len(time_idx)}')

        nc = compute_nc(mat_path, T_START_MS, T_END_MS, NC_SPLITS)
        roi_mask = neural['roi_mask']
        nc_roi   = nc[roi_mask]
        nc_mask2 = nc_roi >= NC_THRESH
        print(f'  NC>=0.4: {nc_mask2.sum()}/{len(nc_roi)}')
        np.save(ses_dir / 'nc.npy', nc_roi)
        np.save(ses_dir / 't_axis.npy', t_ax_win)

        r_ses = {}
        for c in COND_NAMES:
            X = features[c]
            print(f'  [{c}] Ridge ({X.shape[1]} feats → PCA {PCA_DIM})...')
            r_tb = ridge_pearson_r(X, responses, train_idx, time_idx, PCA_DIM)
            np.save(ses_dir / f'r_{c}.npy', r_tb)
            r_ses[c] = r_tb
            r_nc = r_tb[:, nc_mask2] / nc_roi[nc_mask2][None, :]
            e_m = (t_ax_win >= 70) & (t_ax_win <= 120)
            l_m = (t_ax_win >= 120) & (t_ax_win <= 170)
            for u in range(r_nc.shape[1]):
                all_r_psth[c].append(r_nc[:, u])
            all_r_early[c].extend(np.nanmean(r_nc[e_m, :], 0).tolist())
            all_r_late[c].extend(np.nanmean(r_nc[l_m, :], 0).tolist())

        done_flag.touch()
        print(f'  Done in {(time.time()-t0)/60:.1f} min')

        if si == 0 and not session1_saved:
            plot_session1(r_ses, t_ax_win, nc_roi, NC_THRESH, OUTDIR / 'session1_results.png')
            session1_saved = True

    t_plot = np.linspace(T_START_MS, T_END_MS, len(all_r_psth['Full'][0]))
    plot_psth(all_r_psth, t_plot, NC_THRESH, OUTDIR / 'psth_timeseries.png')
    plot_window_bars(all_r_early, all_r_late, NC_THRESH, OUTDIR / 'window_bars.png')
    print('\nAll done.')

if __name__ == '__main__':
    main()
