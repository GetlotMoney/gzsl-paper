# FRAMEWORK-V7 Tune

`V7-TUNE-013_ONE_TEXT_SEEN_ONLY_CE`：CUB一文本Full从正式seed7原始初始化端到端训练，最终head分类CE只在seen logits上计算，推理仍200类竞争。

`V7-TUNE-015_TEXT_RELATION_OPERATOR`：completed/drop；删除逐图Reader的共享identity-residual低秩文本到视觉算子失败，Full H=79.437516，低 formal V7 1.072915 H，停止此方向。
