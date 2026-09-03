# V7-TUNE-013 结果

状态：planned。CUB最小救援实验：相对一文本Full路径，唯一改变是最终head分类CE只在150个seen logits上计算；official推理和评估仍然200类联合竞争。正式V7 `b32a16f` 是准确率地板，不作为“只差一个CE”的直接代码diff基线。CUB方向CE覆盖边界预注册为跳过seen类 `[13,76]`，运行时必须一致。
