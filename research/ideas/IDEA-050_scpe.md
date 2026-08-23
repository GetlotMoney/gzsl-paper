# IDEA-050：Spatially Coherent Patch Evidence

status: testing
problem: CCPE top2只看两个最高patch相似度，两个分散的背景噪声点也可能被当成局部类别证据。
hypothesis: 保持每类top2匹配，但用24×24网格中top2的归一化邻近度加权其平均相似度，可抑制空间分散的伪局部证据并超过CCPE。
evidence_refs: IDEA-049证明top8→top4→top2越尖锐H越高；真实patch物理顺序为24×24网格，IDEA-048表明空间/类别定位不能被平均掉。
base_commit: 06d28c77ee032cd32e81e356797df6f7a5781c29
core_change: 将CCPE的top2均值改为空间邻近度加权top2均值；冻结SEBC父模型，只训练一个beta，不使用人工属性。
success_condition: H大于CCPE最高77.666533，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过CCPE，或beta达到98%上限。
experiment: V2-INNOVATION-016
