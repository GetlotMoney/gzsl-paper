# FRAMEWORK-V7 Ablation

`V7-ABLATION-002_SEEN_CE_RETRAINED_SVI`：completed。TUNE013 CUB seen-only CE 的正式从头重训消融，包含 S-off、V-off、I-off、V+I-off 四组；Full 复用已接纳 TUNE013 正式 RUN H=79.94579718163422。S-off `+0.502084 H` 是唯一接近 0.5H 的稳定贡献；V-off / I-off / V+I-off 仅 `+0.256176 / +0.246076 / +0.246076 H`，且 I-off 与 V+I-off 逐值相同，旧全局 V/I 收掉。
