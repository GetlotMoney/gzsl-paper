# FRAMEWORK-V7 Ablation

`V7-ABLATION-002_SEEN_CE_RETRAINED_SVI`（对应分支 exp/v7/ablation/ablation-002-seen-ce）：TUNE013 seen-only CE 的正式从头重训消融，S-off +0.502084 / V-off +0.256176 / I-off +0.246076 / V+I-off +0.246076 H；旧全局 V/I 收掉。

`V7-ABLATION-003_PURECLIP_SVI`（对应分支 exp/v7/ablation/ablation-003-pureclip-svi）：见其分支账本。

`V7-ABLATION-004_NO_PARENT_HEAD`（本分支）：completed（诊断）。冻结 TG+GTD、head 基于 Mean8 纯文本 base，Full / S-off / V-off / I-off / V+I-off 五组从头重训 + B0 零训练 Mean8 基线（H=68.75）。重训贡献：S +3.77、V +7.22、V+I 联合 +7.51 H；Full=77.78 仍低 formal V7 2.73 H。结论：TG+GTD 遮蔽 S/V/I 假说成立，但 TG+GTD 迁移本身仍有约 2.7H 不可替代增益；弱基线上的涨幅不得当作完整框架模块增益。
