# FRAMEWORK-V7 Tune

`V7-TUNE-013_ONE_TEXT_SEEN_ONLY_CE`：CUB一文本Full从正式seed7原始初始化端到端训练，最终head分类CE只在seen logits上计算，推理仍200类竞争。

`V7-TUNE-014_CLASS_HELD_OUT_VI`：CUB一文本Full的V/I训练改成一阶class-held-out outer CE；TG/GTD/S保持普通seen训练，推理仍200类竞争。
