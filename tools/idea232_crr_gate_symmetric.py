# -*- coding: utf-8 -*-
"""IDEA-232 CRR kill-gate 对称口径正式版 (owner裁决方案a, 2026-09-03).

对称口径: 每类用其作为折外类时该折训练的W修正原型, 竞争场150类全修正 vs 150类全t_c.
消除原口径的竞争场不对称偏置(修正类被拉入图像锥系统性占优, text_pca对照+11.3pp证伪原口径).

判据(预注册点 lambda*=0.1, beta*=0.1):
  c1: 4种子合并均值 delta >= in-run基线+1.0pp
  c2: 每种子3折级中 >=2折为正 (折级=该折50类在全修正场vs全基线场的macro差)
  c3: class-level paired bootstrap(150类, 跨4种子均值差, 10000次) 95%CI下界>0, 仅预注册点
  c4: 最差种子披露, <+0.5pp 标 instability warning (非阻断)
复用主脚本已审查的 load_assets / build_fold_targets / ridge_fit / eval_cba / run_insample.
"""
import argparse
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, '/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/crr/IDEA-232-GATE0')
import idea232_crr_gate as G


def build_symmetric_proto(t, x_full, mu, lam, beta, seed):
    """每类用其作为折外类时该折训练的W修正; 返回全150类修正原型."""
    n_cls = 150
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_cls)
    folds = np.array_split(perm, G.N_FOLD)
    proto = np.zeros_like(t)
    for out_cls in folds:
        out_cls = np.sort(out_cls)
        train_cls = np.setdiff1d(np.arange(n_cls), out_cls)
        y = G.build_fold_targets(t, mu, train_cls)
        xm = x_full[train_cls].mean(0, keepdims=True)
        W = G.ridge_fit(x_full[train_cls] - xm, y, lam)
        rhat = (x_full[out_cls] - xm) @ W
        proto[out_cls] = G.l2r(t[out_cls] + beta * G.l2r(rhat))
    return proto, folds


def eval_seed(t, proto, trf, trl, tsf, tsl, folds):
    """整种子评估: 整体delta + 3折级delta + per-class配对 + test_seen."""
    n_cls = 150
    img_n = G.l2r(trf)
    pred_b = (img_n @ t.T).argmax(1)
    pred_c = (img_n @ proto.T).argmax(1)
    accs_b = np.array([(pred_b[trl == c] == c).mean() for c in range(n_cls)])
    accs_c = np.array([(pred_c[trl == c] == c).mean() for c in range(n_cls)])
    delta = float(accs_c.mean() - accs_b.mean())
    fold_deltas = []
    for out_cls in folds:
        oc = np.sort(out_cls)
        fold_deltas.append(float(accs_c[oc].mean() - accs_b[oc].mean()))
    # test_seen (secondary, 不进判据; 空类跳过)
    ts_n = G.l2r(tsf)
    pred_b2 = (ts_n @ t.T).argmax(1)
    pred_c2 = (ts_n @ proto.T).argmax(1)
    ab2, ac2 = [], []
    for c in range(n_cls):
        m = tsl == c
        if m.sum() == 0:
            continue
        ab2.append((pred_b2[m] == c).mean())
        ac2.append((pred_c2[m] == c).mean())
    return {'delta': delta, 'fold_deltas': fold_deltas,
            'per_class_diff': (accs_c - accs_b), 'base_macro': float(accs_b.mean()),
            'test_seen_delta': float(np.mean(ac2) - np.mean(ab2))}


def run_symmetric(t, x_full, mu, trf, trl, tsf, tsl, lam, beta, seeds=G.SEEDS):
    n_cls = 150
    seed_res = {}
    per_class_acc = []
    base_macro = None
    for seed in seeds:
        proto, folds = build_symmetric_proto(t, x_full, mu, lam, beta, seed)
        r = eval_seed(t, proto, trf, trl, tsf, tsl, folds)
        seed_res[str(seed)] = {'delta_pp': round(r['delta'] * 100, 4),
                               'fold_deltas_pp': [round(d * 100, 4) for d in r['fold_deltas']],
                               'n_pos_folds': sum(1 for d in r['fold_deltas'] if d > 0),
                               'test_seen_delta_pp': round(r['test_seen_delta'] * 100, 4)}
        per_class_acc.append(r['per_class_diff'])
        base_macro = r['base_macro']
    per_class_mean = np.mean(np.stack(per_class_acc), axis=0)
    return {'seed_results': seed_res,
            'mean_delta_pp': round(float(np.mean([v['delta_pp'] for v in seed_res.values()])) * 1.0, 4),
            'in_run_base_pp': round(base_macro * 100, 4),
            'per_class_mean_diff': per_class_mean.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/crr/IDEA-232-GATE0/result_symmetric.json')
    args = ap.parse_args()

    t, q, trf, trl, mu, npc, tsf, tsl, names = G.load_assets()
    result = {
        'idea_id': 'IDEA-232',
        'arena': 'symmetric: all-150 corrected (each class by its out-fold W) vs all-150 t_c baseline',
        'owner_ruling': 'option(a) 2026-09-03: symmetric arena adopted as official metric after asymmetric bias proven (text_pca control +11.3pp ~= attr +12.6pp)',
        'criteria': {'c1': 'mean over 4 seeds >= in-run base +1.0pp',
                     'c2': '>=2/3 fold-deltas positive per seed',
                     'c3': 'class bootstrap 95% CI lower>0 at prereg point only',
                     'c4': 'worst seed disclosed, <+0.5pp instability warning (non-blocking)'},
        'prereg_point': {'lambda': G.LAMBDA_STAR, 'beta': G.BETA_STAR},
        'asset_sha256': {'role_sentence_embeds.pt': G.sha256_of(G.ASSET_DIR + 'role_sentence_embeds.pt'),
                         'train_features.pt': G.sha256_of(G.ASSET_DIR + 'train_features.pt'),
                         'train_labels.pt': G.sha256_of(G.ASSET_DIR + 'train_labels.pt'),
                         'att_splits.mat': G.sha256_of(G.ATT_MAT),
                         'class_names.json': G.sha256_of(G.ASSET_DIR + 'class_names.json'),
                         'test_seen_features.pt': G.sha256_of(G.ASSET_DIR + 'test_seen_features.pt'),
                         'test_seen_labels.pt': G.sha256_of(G.ASSET_DIR + 'test_seen_labels.pt')},
        'contract': {'seeds': G.SEEDS, 'n_fold': G.N_FOLD},
    }

    # ---- 主条件: 预注册点 ----
    main = run_symmetric(t, q, mu, trf, trl, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR)
    result['main'] = main

    verdict = {}
    verdict['c1_mean_ge_1pp'] = bool(main['mean_delta_pp'] >= G.GATE_PP)
    verdict['c2_seed_2of3_folds_pos'] = all(v['n_pos_folds'] >= 2 for v in main['seed_results'].values())
    pc = np.array(main['per_class_mean_diff'])
    rng = np.random.RandomState(0)
    boots = [float(pc[rng.randint(0, 150, 150)].mean()) for _ in range(G.BOOTSTRAP_N)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    verdict['c3_bootstrap_ci'] = {'low_pp': round(lo * 100, 4), 'high_pp': round(hi * 100, 4), 'pass': bool(lo > 0)}
    worst = min(v['delta_pp'] for v in main['seed_results'].values())
    verdict['c4_worst_seed'] = {'delta_pp': worst, 'instability_warning': bool(worst < 0.5)}
    result['verdict'] = verdict
    result['gate_pass'] = bool(verdict['c1_mean_ge_1pp'] and verdict['c2_seed_2of3_folds_pos'] and verdict['c3_bootstrap_ci']['pass'])

    # ---- 预注册对照(对称口径, 同折同W流程) ----
    ctrl = {}
    sh = []
    for s in G.SHUFFLE_SEEDS:
        rs = np.random.RandomState(s)
        x_sh = q[rs.permutation(150)]
        sh.append(run_symmetric(t, x_sh, mu, trf, trl, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR)['mean_delta_pp'])
    ctrl['shuffled_attr'] = {'mean_deltas_pp': sh}
    x_freq = npc[:, None]
    ctrl['class_freq'] = {'mean_delta_pp': run_symmetric(t, x_freq, mu, trf, trl, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR)['mean_delta_pp']}
    # text PCA (全150类拟合->~149维; 折内PCA无法进入symmetric builder接口, 对照披露口径)
    cen = t - t.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(cen, full_matrices=False)
    tx_full = cen @ Vt[:312].T
    ctrl['text_pca_fullfit(~149d)'] = {'mean_delta_pp': run_symmetric(t, tx_full, mu, trf, trl, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR)['mean_delta_pp'],
                                       'note': 'full-150 PCA fit, fold-PCA not reachable inside symmetric builder; control only'}
    # oracle上限: 真实r_c全150类注入
    allr = mu - (mu * t).sum(-1, keepdims=True) * t
    allr = allr - allr.mean(0, keepdims=True)
    oracle_proto = G.l2r(t + G.BETA_STAR * G.l2r(allr))
    img_n = G.l2r(trf)
    pb = (img_n @ t.T).argmax(1); pc_ = (img_n @ oracle_proto.T).argmax(1)
    ab = np.array([(pb[trl == c] == c).mean() for c in range(150)])
    ac = np.array([(pc_[trl == c] == c).mean() for c in range(150)])
    ctrl['oracle_r_upper'] = {'mean_delta_pp': round(float(ac.mean() - ab.mean()) * 100, 4)}
    pm = (img_n @ mu.T).argmax(1)
    am = np.array([(pm[trl == c] == c).mean() for c in range(150)])
    ctrl['mu_upper'] = {'mean_delta_pp': round(float(am.mean() - ab.mean()) * 100, 4)}
    # in-sample披露(复用主脚本: 全150拟合W全修正)
    ctrl['insample_disclosure'] = G.run_insample(t, q, trf, trl, mu, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR)
    result['controls'] = ctrl

    # ---- 网格稳定性披露(seed=7单种子, 同原合同口径) ----
    grid = {}
    for lam in G.LAMBDA_GRID:
        for beta in G.BETA_GRID:
            grid[f'lam{lam}_beta{beta}'] = run_symmetric(t, q, mu, trf, trl, tsf, tsl, lam, beta, seeds=[7])['mean_delta_pp']
    n_pass = sum(1 for v in grid.values() if v >= 0.5)
    result['grid_disclosure'] = {'seed': 7, 'values_pp': grid, 'n_ge_0.5pp_of_total': f'{n_pass}/{len(grid)}'}

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'gate_pass': result['gate_pass'],
                      'mean_delta_pp': main['mean_delta_pp'],
                      'in_run_base_pp': main['in_run_base_pp'],
                      'verdict': verdict,
                      'controls': {k: v for k, v in ctrl.items() if k != 'insample_disclosure'},
                      'insample': ctrl['insample_disclosure'],
                      'grid_n_ge_0.5pp': result['grid_disclosure']['n_ge_0.5pp_of_total']},
                     ensure_ascii=False, indent=2))
    print('saved:', args.out)


if __name__ == '__main__':
    main()
