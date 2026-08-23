# IDEA-057：Local Visual Prototype Generation

status: rejected
problem: CCPE直接用文本残差匹配CLIP patch，文本与局部视觉特征仍存在模态间隙；其后的标量校准均无增益。
hypothesis: 用seen真类局部文本从每张训练图选择top2 patch，形成150类局部视觉中心，再以ridge从局部语义生成200类视觉化局部原型，可缩小patch-文本模态间隙并超过CCPE。
evidence_refs: IDEA-049证明top2局部定位有效；IDEA-056说明只调类别权重无效；IDEA-016的全局SVPG曾失败，因此本次仅在文本定位后的局部patch空间拟合，并保留独立止损边界。
base_commit: 8dd45d33ad1fee24c910590293adb951838f177f
core_change: 把CCPE的局部文本原型替换为seen局部视觉中心ridge生成的200类局部视觉原型；推理仍为每类top2 patch证据，只训练一个beta。
success_condition: H大于CCPE最高77.666533，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过CCPE，或beta达到98%上限。
experiment: V2-INNOVATION-023
result: beta饱和但所有非零条件均降低H，best退回SEBC；seen局部视觉ridge仍产生unseen域偏置。
