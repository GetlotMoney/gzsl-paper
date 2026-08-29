# IDEA-075：Sentence-Dropout Conservative Routing

status: supported
problem: CASR全局权重可靠，但训练可能依赖少数高权重句子；图像门控和类别路由均无效，需要不增加推理模块的稳健训练目标。
hypothesis: 从CASR精确起步，训练时每批随机屏蔽一句并以0.01 KL限制完整权重，可提高八句冗余稳健性；推理恢复完整8句并超过CASR。
evidence_refs: IDEA-072 CASR两seed可靠；IDEA-074动态门控退化；句子dropout只改变训练目标，不增加推理复杂度。
base_commit: 561dfc4f4e279556ea047dc610adf422ca199d9b
core_change: 固定CASR beta，从CASR全局句权重起步训练±0.5残差；每个训练batch随机mask一句，推理使用完整8句。
success_condition: H大于CASR最高78.285719，U和S任一项下降不超过2个百分点，推理权重std大于0.01且min大于0.01。
failure_condition: H不超过CASR或推理权重塌缩。
experiment: V2-INNOVATION-041
result: seed5/7 H=78.320510/78.302856，两链都超过CASR且权重非塌缩、mask覆盖均衡；可靠成立，最高取seed5。
