# -*- coding: utf-8 -*-
"""IDEA-232 对称口径补充检查: 竞争场不对称偏置诊断.

原Gate口径: 折外50类修正 vs 折内100类保持t_c -> 修正类系统性分数偏置(伪增益).
对称口径: 每类用其作为折外类时该折训练的W修正, 150类全修正 vs 150类全t_c.
不修改主脚本与已产生结果; 本检查作为口径缺陷证据回填.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/crr/IDEA-232-GATE0')
import idea232_crr_gate as G


def symmetric_oof(t, q, trf, trl, mu, n_per_class, tsf, tsl, lam, beta, feature='attr',
                  shuffle_seed=None, seeds=G.SEEDS):
    n_cls = 150
    per_class = {c: [] for c in range(n_cls)}
    base_all, crr_all = [], []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_cls)
        folds = np.array_split(perm, N_FOLD := 3)
        proto = np.zeros_like(t)
        for out_cls in folds:
            out_cls = np.sort(out_cls)
            train_cls = np.setdiff1d(np.arange(n_cls), out_cls)
            y = G.build_fold_targets(t, mu, train_cls)
            if feature == 'attr':
                x_full = q.copy()
                if shuffle_seed is not None:
                    rs = np.random.RandomState(shuffle_seed)
                    x_full = x_full[rs.permutation(n_cls)]
            elif feature == 'freq':
                x_full = n_per_class[:, None].copy()
            elif feature == 'text312':
                tin = t[train_cls]
                cen = tin - tin.mean(0, keepdims=True)
                U, S, Vt = np.linalg.svd(cen, full_matrices=False)
                V = Vt[:312].T
                x_full = (t - tin.mean(0, keepdims=True)) @ V
            xm = x_full[train_cls].mean(0, keepdims=True)
            W = G.ridge_fit(x_full[train_cls] - xm, y, lam)
            rhat = (x_full[out_cls] - xm) @ W
            rr = G.l2r(rhat)
            proto[out_cls] = G.l2r(t[out_cls] + beta * rr)
        # 全150类图像, 竞争场: 全t_c vs 全修正
        b, accs_b = G.eval_cba(trf, trl, t, list(range(n_cls)))
        c, accs_c = G.eval_cba(trf, trl, proto, list(range(n_cls)))
        base_all.append(b); crr_all.append(c)
        for cc in range(n_cls):
            per_class[cc].append(accs_c[cc] - accs_b[cc])
    per_class_mean = np.array([np.mean(v) for v in per_class.values()])
    rng = np.random.RandomState(0)
    boots = [per_class_mean[rng.randint(0, 150, 150)].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # test_seen 同口径
    b2, _ = G.eval_cba(tsf, tsl, t, list(range(n_cls)))
    c2, _ = G.eval_cba(tsf, tsl, proto, list(range(n_cls)))
    return {'mean_delta_pp': float(np.mean(crr_all) - np.mean(base_all)) * 100,
            'base_pp': float(np.mean(base_all)) * 100, 'crr_pp': float(np.mean(crr_all)) * 100,
            'bootstrap_ci_pp': [float(lo) * 100, float(hi) * 100],
            'test_seen_delta_pp': (c2 - b2) * 100}


def main():
    t, q, trf, trl, mu, npc, tsf, tsl, names = G.load_assets()
    out = {}
    out['symmetric_attr'] = symmetric_oof(t, q, trf, trl, mu, npc, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR)
    out['symmetric_text312'] = symmetric_oof(t, q, trf, trl, mu, npc, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR, feature='text312')
    out['symmetric_freq'] = symmetric_oof(t, q, trf, trl, mu, npc, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR, feature='freq')
    sh = [symmetric_oof(t, q, trf, trl, mu, npc, tsf, tsl, G.LAMBDA_STAR, G.BETA_STAR, shuffle_seed=s)['mean_delta_pp'] for s in [101, 102, 103]]
    out['symmetric_shuffled_attr'] = sh
    # beta敏感性(对称口径)
    out['symmetric_beta_scan'] = {str(b): symmetric_oof(t, q, trf, trl, mu, npc, tsf, tsl, G.LAMBDA_STAR, b)['mean_delta_pp'] for b in [0.05, 0.1, 0.15, 0.2, 0.3]}
    with open('/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/crr/IDEA-232-GATE0/symmetric_check.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
