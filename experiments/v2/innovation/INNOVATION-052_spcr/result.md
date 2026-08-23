# V2-INNOVATION-052 结果

状态：`rejected_inference_conflict`。

RUN-001从小patch beta到接近`5`边界的完整轨迹均低于SDCR，best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`、selected iteration=`-1`、patch beta=`0`。小beta约`0.28`时H已降到`78.276210%`，排除“范围过宽掩盖微小最优”的可能。

CCPE top2在SEBC父链有效，但直接叠加到SDCR推理logits会破坏已形成的全局句子平衡。IDEA-086拒绝，不做beta补救；后续若使用patch，只允许作为训练期可靠性信息，推理不再增加patch分支。

模型SHA256：`4de3727c4b449ca31dba0e88c1e793b6b06118dc1adc08a8e795a24273b4b745`；最后checkpoint SHA256：`f60349a790bc6bca4ba3217a70e482c790d7e7a9d68dfe8937ff6853535d81eb`。
