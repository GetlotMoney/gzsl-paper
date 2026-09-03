# -*- coding: utf-8 -*-
"""IDEA-232 CRR kill-gate: 专家属性 -> 类视觉残差闭式ridge回归 OOF Gate.

冻结预注册合同见 research/ideas/IDEA-232_class_level_residual_regression.md。
唯一实验变量: 折外类原型是否加预测残差 p_c = L2(t_c + beta*L2(x_c W))。
主口径 trainval 折外类图像 / 150类竞争 / CBA=折外类macro accuracy, 不碰test。
secondary 披露 test_seen (不进判据)。
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import scipy.io as sio
import torch

ASSET_DIR = '/data/lby/projects/cv_project/GZSL_Warehouse/assets/clip_vitl14_336/CUB/69c9c6d82a755fe8/'
ATT_MAT = '/data/lby/projects/cv_project/GZSL_Warehouse/datasets/splits/xlsa17/data/CUB/att_splits.mat'

SEEDS = [7, 11, 33, 55]
N_FOLD = 3
LAMBDA_STAR = 0.1
BETA_STAR = 0.1
LAMBDA_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
BETA_GRID = [0.05, 0.1, 0.15, 0.2, 0.3]
SHUFFLE_SEEDS = [101, 102, 103, 104, 105]
GATE_PP = 1.0
BOOTSTRAP_N = 10000


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def l2r(a):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def load_assets():
    emb = torch.load(ASSET_DIR + 'role_sentence_embeds.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
    assert emb.shape == (200, 8, 768), emb.shape
    # t_c: 句子直接均值 -> 行L2, 禁止句子预归一化
    t = l2r(emb.mean(axis=1))  # [200,768]

    trf = torch.load(ASSET_DIR + 'train_features.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
    trl = torch.load(ASSET_DIR + 'train_labels.pt', map_location='cpu', weights_only=True).numpy()
    assert trf.shape == (7057, 768) and trl.min() == 0 and trl.max() == 149, (trf.shape, trl.min(), trl.max())

    cn = json.load(open(ASSET_DIR + 'class_names.json'))
    asset_names = cn['xlsa']  # 前150=seen, 后50=unseen (与trainval/test_unseen loc一致)
    assert len(asset_names) == 200

    m = sio.loadmat(ATT_MAT)
    att = m['att'].astype(np.float64)  # [312,200] xlsa17原始顺序
    xlsa_names = [x[0][0] for x in m['allclasses_names']]
    assert len(xlsa_names) == 200
    # 对齐: att列(xlsa顺序) -> asset顺序
    xlsa_idx = {n: i for i, n in enumerate(xlsa_names)}
    unmatched = [n for n in asset_names if n not in xlsa_idx]
    print(f'[align] matched {len(asset_names) - len(unmatched)}/200; unmatched: {unmatched}')
    if unmatched:
        raise RuntimeError(f'class alignment failed, {len(unmatched)} unmatched: {unmatched[:5]}')
    perm = [xlsa_idx[n] for n in asset_names]
    q = att[:, perm].T  # [200,312] 行序与role_sentence_embeds一致

    # 类视觉中心 mu_c = L2(mean(raw train features per class)), 先均值后归一化
    mu = np.zeros((200, 768))
    n_per_class = np.zeros(200)
    for c in range(150):
        feats = trf[trl == c]
        n_per_class[c] = len(feats)
        mu[c] = l2r(feats.mean(axis=0))
    assert (n_per_class[:150] > 0).all() and (n_per_class[150:] == 0).all()

    tsf = torch.load(ASSET_DIR + 'test_seen_features.pt', map_location='cpu', weights_only=True).numpy().astype(np.float64)
    tsl = torch.load(ASSET_DIR + 'test_seen_labels.pt', map_location='cpu', weights_only=True).numpy()
    assert tsf.shape[0] == 1764 and tsl.min() == 0 and tsl.max() == 149, (tsf.shape, tsl.min(), tsl.max())

    return t, q, trf, trl, mu, n_per_class, tsf, tsl, asset_names


def build_fold_targets(t, mu, train_cls):
    """折内类: r_c = (I - t_c t_c^T) mu_c -> 折内中心化 -> 单位方向."""
    tc = t[train_cls]                       # [n,768]
    muc = mu[train_cls]                     # [n,768]
    r = muc - (muc * tc).sum(-1, keepdims=True) * tc  # 去t_c平行分量
    r = r - r.mean(0, keepdims=True)        # 折内中心化(去模态间隙共有分量)
    nrm = np.linalg.norm(r, axis=-1, keepdims=True)
    y = r / np.maximum(nrm, 1e-12)          # 单位方向目标
    return y


def ridge_fit(x, y, lam):
    d = x.shape[1]
    return np.linalg.solve(x.T @ x + lam * np.eye(d), x.T @ y)


def apply_prototypes(t_all, out_cls, base_proto, beta, resid_hat):
    """折外类原型 = L2(t_c + beta*L2(rhat_c)); 竞争场其余类保持基线原型."""
    proto = base_proto.copy()
    rr = l2r(resid_hat)
    proto[out_cls] = l2r(t_all[out_cls] + beta * rr)
    return proto


def eval_cba(img_feats, img_labels, proto, eval_cls, img_norm=None):
    """eval_cls 的 macro accuracy; 竞争场=proto全部行."""
    if img_norm is None:
        img_norm = l2r(img_feats)
    sims = img_norm @ proto.T  # [N,C]
    pred = sims.argmax(1)
    accs = []
    for c in eval_cls:
        mask = img_labels == c
        if mask.sum() == 0:
            continue
        accs.append((pred[mask] == c).mean())
    return float(np.mean(accs)), accs


def run_oof(t, q, trf, trl, mu, n_per_class, tsf, tsl, lam, beta, feature='attr',
            shuffle_seed=None, seeds=SEEDS):
    """返回: 每种子折级delta列表, 类级配对acc差[150], 以及每折明细."""
    n_cls = 150
    fold_deltas = []          # [(seed, fold, d_trainval, d_testseen)]
    per_class_trainval = {c: [] for c in range(n_cls)}
    base_tr, crr_tr = [], []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_cls)
        folds = np.array_split(perm, N_FOLD)
        for fi, out_cls in enumerate(folds):
            out_cls = np.sort(out_cls)
            train_cls = np.setdiff1d(np.arange(n_cls), out_cls)
            # 折内目标
            y = build_fold_targets(t, mu, train_cls)
            # 折内特征
            if feature == 'attr':
                x_full = q[:n_cls].copy()
                if shuffle_seed is not None:
                    rs = np.random.RandomState(shuffle_seed)
                    x_full = x_full[rs.permutation(n_cls)]
            elif feature == 'freq':
                x_full = n_per_class[:n_cls, None].copy()  # 类频率单特征
            elif feature == 'text312':
                # t_c PCA->至多312维(折内100类秩上限99,实际~100维), 拟合只用折内类
                tin = t[train_cls]
                cen = tin - tin.mean(0, keepdims=True)
                U, S, Vt = np.linalg.svd(cen, full_matrices=False)
                V = Vt[:312].T  # [768, min(312,rank)]
                x_full = (t[:n_cls] - tin.mean(0, keepdims=True)) @ V
            else:
                raise ValueError(feature)
            xm = x_full[train_cls].mean(0, keepdims=True)
            x_tr = x_full[train_cls] - xm
            W = ridge_fit(x_tr, y, lam)
            # 折外类预测残差
            rhat = (x_full[out_cls] - xm) @ W
            # 竞争场: 基线原型全部=t_c; CRR仅折外50类替换
            base_proto = t[:n_cls]
            crr_proto = apply_prototypes(t, out_cls, base_proto, beta, rhat)
            # trainval 折外类图像
            imgs, labs = [], []
            for c in out_cls:
                m = trl == c
                imgs.append(trf[m]); labs.append(trl[m])
            imgs = np.concatenate(imgs); labs = np.concatenate(labs)
            b, accs_b = eval_cba(imgs, labs, base_proto, out_cls)
            c, accs_c = eval_cba(imgs, labs, crr_proto, out_cls)
            fold_deltas.append({'seed': seed, 'fold': fi,
                                'base_trainval': b, 'crr_trainval': c, 'delta_trainval': c - b})
            for cc, ab, ac in zip(out_cls, accs_b, accs_c):
                per_class_trainval[cc].append(ac - ab)
            base_tr.append(b); crr_tr.append(c)
            # secondary: test_seen 折外类图像(不进判据)
            imgs2, labs2 = [], []
            for c in out_cls:
                m2 = tsl == c
                if m2.sum() > 0:
                    imgs2.append(tsf[m2]); labs2.append(tsl[m2])
            imgs2 = np.concatenate(imgs2); labs2 = np.concatenate(labs2)
            b2, _ = eval_cba(imgs2, labs2, base_proto, out_cls)
            c2, _ = eval_cba(imgs2, labs2, crr_proto, out_cls)
            fold_deltas[-1].update({'base_testseen': b2, 'crr_testseen': c2, 'delta_testseen': c2 - b2})
    per_class_mean = np.array([np.mean(per_class_trainval[c]) for c in range(n_cls)])
    summary = {
        'mean_delta_trainval': float(np.mean([f['delta_trainval'] for f in fold_deltas])),
        'mean_base_trainval': float(np.mean(base_tr)),
        'mean_crr_trainval': float(np.mean(crr_tr)),
        'mean_delta_testseen': float(np.mean([f['delta_testseen'] for f in fold_deltas])),
        'seed_means': {},
        'fold_details': fold_deltas,
        'per_class_delta_mean_trainval': per_class_mean.tolist(),
    }
    for s in seeds:
        ds = [f['delta_trainval'] for f in fold_deltas if f['seed'] == s]
        pos = sum(1 for d in ds if d > 0)
        summary['seed_means'][str(s)] = {'mean': float(np.mean(ds)), 'n_pos_folds': pos, 'n_folds': len(ds)}
    return summary, per_class_mean


def run_oracle(t, trf, trl, mu, tsf, tsl, beta, seeds=SEEDS):
    """上限锚点: 折外类原型加真实r_c (oracle) 或直接=mu_c."""
    out = {}
    for mode in ['oracle', 'mu']:
        deltas = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(150)
            folds = np.array_split(perm, N_FOLD)
            for out_cls in folds:
                out_cls = np.sort(out_cls)
                base_proto = t[:150]
                proto = base_proto.copy()
                if mode == 'oracle':
                    tc = t[out_cls]; muc = mu[out_cls]
                    # oracle残差: 全局中心化(150类)后的r_c, 与管线一致用类均值中心化
                    allr = mu[:150] - (mu[:150] * t[:150]).sum(-1, keepdims=True) * t[:150]
                    r = allr - allr.mean(0, keepdims=True)
                    rr = r[out_cls] / np.maximum(np.linalg.norm(r[out_cls], axis=-1, keepdims=True), 1e-12)
                    proto[out_cls] = l2r(tc + beta * rr)
                else:
                    proto[out_cls] = mu[out_cls]
                imgs, labs = [], []
                for c in out_cls:
                    m = trl == c
                    imgs.append(trf[m]); labs.append(trl[m])
                imgs = np.concatenate(imgs); labs = np.concatenate(labs)
                b, _ = eval_cba(imgs, labs, base_proto, out_cls)
                c, _ = eval_cba(imgs, labs, proto, out_cls)
                deltas.append(c - b)
        out[mode] = {'mean_delta_trainval': float(np.mean(deltas))}
    return out


def run_insample(t, q, trf, trl, mu, tsf, tsl, lam, beta):
    """披露: 全拟合(150类训W, 全部150类修正) -> 量化记忆成分."""
    y = build_fold_targets(t, mu, np.arange(150))
    x = q[:150] - q[:150].mean(0, keepdims=True)
    W = ridge_fit(x, y, lam)
    rhat = (q[:150] - q[:150].mean(0, keepdims=True)) @ W
    proto = l2r(t[:150] + beta * l2r(rhat))
    b_tr, _ = eval_cba(trf, trl, t[:150], list(range(150)))
    c_tr, _ = eval_cba(trf, trl, proto, list(range(150)))
    b_ts, _ = eval_cba(tsf, tsl, t[:150], list(range(150)))
    c_ts, _ = eval_cba(tsf, tsl, proto, list(range(150)))
    return {'trainval_base': b_tr, 'trainval_insample': c_tr, 'trainval_delta': c_tr - b_tr,
            'test_seen_base': b_ts, 'test_seen_insample': c_ts, 'test_seen_delta': c_ts - b_ts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/idea232_gate_result.json')
    args = ap.parse_args()

    t, q, trf, trl, mu, npc, tsf, tsl, names = load_assets()
    result = {
        'idea_id': 'IDEA-232',
        'asset_sha256': {
            'role_sentence_embeds.pt': sha256_of(ASSET_DIR + 'role_sentence_embeds.pt'),
            'train_features.pt': sha256_of(ASSET_DIR + 'train_features.pt'),
            'train_labels.pt': sha256_of(ASSET_DIR + 'train_labels.pt'),
            'att_splits.mat': sha256_of(ATT_MAT),
            'class_names.json': sha256_of(ASSET_DIR + 'class_names.json'),
            'test_seen_features.pt': sha256_of(ASSET_DIR + 'test_seen_features.pt'),
            'test_seen_labels.pt': sha256_of(ASSET_DIR + 'test_seen_labels.pt'),
        },
        'contract': {'seeds': SEEDS, 'n_fold': N_FOLD, 'lambda_star': LAMBDA_STAR,
                     'beta_star': BETA_STAR, 'gate_pp': GATE_PP, 'main_metric': 'trainval OOF macro CBA, 150-class arena, out-fold classes only'},
    }

    # ---- 主条件: 预注册点 ----
    main_summary, per_class = run_oof(t, q, trf, trl, mu, npc, tsf, tsl, LAMBDA_STAR, BETA_STAR)
    result['main'] = main_summary

    # ---- 判据 ----
    verdict = {}
    verdict['c1_mean_ge_1pp'] = main_summary['mean_delta_trainval'] >= GATE_PP / 100.0
    verdict['c2_seed_2of3_pos'] = all(v['n_pos_folds'] >= 2 for v in main_summary['seed_means'].values())
    rng = np.random.RandomState(0)
    n = len(per_class)
    boots = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.randint(0, n, n)
        boots.append(per_class[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    verdict['c3_bootstrap_ci'] = {'low': float(lo), 'high': float(hi), 'pass': bool(lo > 0)}
    worst = min(v['mean'] for v in main_summary['seed_means'].values())
    verdict['c4_worst_seed'] = {'mean_delta': worst, 'instability_warning': bool(worst < 0.005)}
    result['verdict'] = verdict

    # ---- 网格稳定性披露 ----
    grid = {}
    for lam in LAMBDA_GRID:
        for beta in BETA_GRID:
            s, _ = run_oof(t, q, trf, trl, mu, npc, tsf, tsl, lam, beta, seeds=[7])
            grid[f'lam{lam}_beta{beta}'] = round(s['mean_delta_trainval'], 6)
    n_pass = sum(1 for v in grid.values() if v >= 0.005)
    result['grid_disclosure'] = {'values_pp': {k: round(v * 100, 3) for k, v in grid.items()},
                                 'n_ge_0.5pp_of_total': f'{n_pass}/{len(grid)}'}

    # ---- 预注册对照 ----
    ctrl = {}
    sh = [run_oof(t, q, trf, trl, mu, npc, tsf, tsl, LAMBDA_STAR, BETA_STAR, shuffle_seed=s)[0]
          for s in SHUFFLE_SEEDS]
    ctrl['shuffled_attr'] = {'mean_deltas_pp': [round(s['mean_delta_trainval'] * 100, 3) for s in sh]}
    fq, _ = run_oof(t, q, trf, trl, mu, npc, tsf, tsl, LAMBDA_STAR, BETA_STAR, feature='freq')
    ctrl['class_freq'] = {'mean_delta_pp': round(fq['mean_delta_trainval'] * 100, 3)}
    tx, _ = run_oof(t, q, trf, trl, mu, npc, tsf, tsl, LAMBDA_STAR, BETA_STAR, feature='text312')
    ctrl['text_pca_infold(actual_dim~100)'] = {'mean_delta_pp': round(tx['mean_delta_trainval'] * 100, 3),
                                               'note': 'fold-PCA rank capped at 99/100, contract 312 not reachable without leakage'}
    ctrl['upper_bounds'] = run_oracle(t, trf, trl, mu, tsf, tsl, BETA_STAR)
    ctrl['insample_disclosure'] = run_insample(t, q, trf, trl, mu, tsf, tsl, LAMBDA_STAR, BETA_STAR)
    result['controls'] = ctrl

    overall_pass = verdict['c1_mean_ge_1pp'] and verdict['c2_seed_2of3_pos'] and verdict['c3_bootstrap_ci']['pass']
    result['gate_pass'] = bool(overall_pass)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'gate_pass': overall_pass, 'verdict': verdict,
                      'mean_delta_pp': round(main_summary['mean_delta_trainval'] * 100, 3),
                      'controls_pp': {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if 'delta' in str(kk) or 'mean' in str(kk)}) for k, v in ctrl.items()}},
                     ensure_ascii=False, indent=2, default=str))
    print('saved:', args.out)


if __name__ == '__main__':
    main()
