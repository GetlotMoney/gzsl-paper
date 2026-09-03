# FRAMEWORK-V7 Tune

`V7-TUNE-013_ONE_TEXT_SEEN_ONLY_CE`：CUB一文本Full从正式seed7原始初始化端到端训练，最终head分类CE只在seen logits上计算，推理仍200类竞争。

`V7-TUNE-014_CLASS_HELD_OUT_VI`：completed/drop；一阶class-held-out outer CE 把 V/I 训练信号改为类别留出迁移，Full H=80.138486，相对 TUNE013 +0.192688 H、U/S 更平衡，但仍低 formal V7 0.371946 H，仅作训练机制候选。
