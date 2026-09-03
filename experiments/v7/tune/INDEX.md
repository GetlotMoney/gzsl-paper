# FRAMEWORK-V7 Tune

`V7-TUNE-013_ONE_TEXT_SEEN_ONLY_CE`：CUB一文本Full从正式seed7原始初始化端到端训练，最终head分类CE只在seen logits上计算，推理仍200类竞争。

`V7-TUNE-015_TEXT_RELATION_OPERATOR`：CUB一文本关系删除逐图Reader，学习共享identity-residual低秩文本关系到视觉中心差分算子，并导出单个`q,bias`分类矩阵。
