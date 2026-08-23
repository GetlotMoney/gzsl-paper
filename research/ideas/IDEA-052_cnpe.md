# IDEA-052：Class-Normalized Patch Evidence

status: testing
problem: MPPE说明局部文本很容易在无关图像中找到高分patch；CCPE原始top2分数也包含不同文本原型固有的公共高分偏置。
hypothesis: 用7,057张seen训练图像估计每个类别文本的top2 patch分数均值和标准差，把train/test分数转为相对seen参考分布的z-score，可保留异常类别特异证据并抑制公共羽色/背景偏置。
evidence_refs: IDEA-049证明top2类别条件patch有效；IDEA-051证明原始最大相似度会累积公共伪匹配；归一化统计只使用seen训练图像且不需要标签。
base_commit: a6d27bc782963353a704a1c6bf4a9301e794962e
core_change: 在CCPE top2分数后增加按类别的seen参考均值/标准差归一化；冻结SEBC父模型，只训练一个beta。
success_condition: H大于CCPE最高77.666533，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过CCPE，或beta达到98%上限。
experiment: V2-INNOVATION-018
