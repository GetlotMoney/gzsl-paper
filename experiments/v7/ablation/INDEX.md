# FRAMEWORK-V7 Ablation

`V7-ABLATION-002_SEEN_CE_RETRAINED_SVI`（对应分支 exp/v7/ablation/ablation-002-seen-ce）：TUNE013 seen-only CE 的正式从头重训消融，S-off +0.502084 / V-off +0.256176 / I-off +0.246076 / V+I-off +0.246076 H；旧全局 V/I 收掉。

`V7-ABLATION-003_PURECLIP_SVI`（对应分支 exp/v7/ablation/ablation-003-pureclip-svi）：见其分支账本。

`V7-ABLATION-004_NO_PARENT_HEAD`（本分支）：去父重训消融（planned）。冻结 TG+GTD source、head 基于 Mean8 纯文本 base，Full / S-off / V-off / I-off / V+I-off 五组从头重训 + B0 零训练 Mean8 基线；用于判定“TG+GTD 遮蔽 V/I” vs “CLS+一文本语义路径到顶”。
