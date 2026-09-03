# V7-ABLATION-002 结果

状态：completed。TUNE013 CUB seen-only CE 的正式从头重训消融，不复用 off checkpoint，也不把同 checkpoint inference-off 诊断当作模块消融。

Full reference 复用已接纳 TUNE013 正式 RUN：H=79.94579718163422，code commit `35cefc52896c383e1ec75a3adc5f78d218d616a3`，config SHA `cadceb65cb9596ea8f9769424154636938120691f306766c8c769168d1f711b3`，metrics SHA `08fd38f284eced2ccd4ea49e1faae1f1163ce5b3821eb7373f9acedf9201f34e`，model SHA `7349e869c1ffef41c3cfe265486880a1c455feb442ff326ae8403bd9a9a39328`。

四个条件均从 seed7 原始初始化重新训练 28,228 updates（code commit `ed2655e425478648a90fad6a84971e9b53583f71`），每 141 update official test 评估一次，各自按本条件 H 选择 best checkpoint。U/S/H 使用 200 类联合竞争，ZS 使用 50 个 unseen 类竞争；trainval seen 图像产生梯度，unseen 图像不产生梯度。

## 各条件最佳结果（best-H 对应同一 checkpoint）

| condition | U | S | H | ZS | best_update | Full−condition H |
|---|---|---|---|---|---|---|
| Full（TUNE013） | 79.960012 | 79.931587 | 79.945797 | 87.821192 | 13818 | — |
| S-off | 78.374481 | 80.542523 | 79.443713 | 86.768633 | 15792 | +0.502084 |
| V-off | 77.788538 | 81.685954 | 79.689621 | 88.308257 | 20586 | +0.256176 |
| I-off | 80.920839 | 78.514910 | 79.699721 | 88.105041 | 18330 | +0.246076 |
| V+I-off | 80.920839 | 78.514910 | 79.699721 | 88.105041 | 18330 | +0.246076 |

独立 best-ZS 观察（非 best-H 同一 checkpoint，仅供查看，不跨 checkpoint 拼接）：S-off ZS=87.376076@18048；V-off ZS=88.347304@13818；I-off 与 V+I-off ZS=88.401026@21291。

## 判断

- S 有稳定小正贡献约 `+0.502084 H`（U 掉 1.58、S 掉 0.61、ZS 掉 1.05），S 是一文本 seen-only CE 路径中唯一接近 0.5H 的贡献来源。
- 旧全局 V/I 联合贡献偏小且不稳定：V-off `+0.256176 H`、I-off `+0.246076 H`、V+I-off `+0.246076 H`，均不足 1H，不能作为核心创新。
- I-off 与 V+I-off 的最佳 U/S/H/ZS 与 best_update 逐值完全相同，说明 I 关闭后 Reader 即使被方向 CE 训练也不影响最终分类（I 的可迁移信息接近 0）。
- 结论：旧 V/I 收掉，不再调 alpha/gamma/Reader 层数；V/I 需引入与当前全局 CLS、旧 CLIP patch 都不同的新学习信号或新路径才值得继续。

## 产物

- output：`/data/lby/projects/cv_project/GZSL_Warehouse/ablation/v7/V7-ABLATION-002_SEEN_CE_RETRAINED_SVI/RUN-{S,V,I,VI}-OFF`
- metrics SHA：S-off `07a5fba0bd06adf6d18117684127b88a6952dc117e1503b48446d88231f09540`；V-off `27f2473eb3b69b96795033a0598278484a5d82e6d34bfe2e944d6a9cd78920de`；I-off `0189844cbe0bb3dc1cb3696bfe43221cd5f537e8cf42381a46a7d047ffd34350`；V+I-off `19589394f88275ce35a6872d1bdb28fb240b83195e61a450648037570718258e`
- model SHA：S-off `f0585e66716dfdf290624038b2f77578e5d6f6379f4a1fd579dfbc9f99616929`；V-off `fcbb9a1c7757ac9050f865ac874c8aea82223aff2a0f35918f031d1bd5e01ecf`；I-off `b80a321b7a9a488c17c9075a798dd44dffcc9a82fb44129a5cac90d7478d9186`；V+I-off `9b6c2496a559b3ac07f4b332774b7aee497af7e3c047a6094473e0aa7af47d29`
- history SHA：S-off `faa04fd69f16998ad9cc48e16d2546377f1d05880a3dd8c658c87f19c4573501`；V-off `a6e150e524119badd790447cde8c9d55655ea6b59beece16ec6246770f9bb4fd`；I-off `565ceb138dae5ddb06e75b9ccef2dd0152ff0bad1b9ed10015be90ed90627d86`；V+I-off `9df43bf8b288ab84787ea0e9e2ba872e52cd061618cd1355167962ce88264e72`
