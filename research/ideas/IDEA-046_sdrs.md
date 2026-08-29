# IDEA-046：Semantic-Disagreement Residual Scaling

status: supported
problem: NCRA让全部200类共享同一个类名残差beta，但GPT长描述与直接类名的一致程度因类别而异。
hypothesis: 以父原型和类名原型的余弦分歧为类别条件，在NCRA最优beta附近学习一个小幅斜率，可以让证据冲突大的类别获得不同修正并提高H。
evidence_refs: IDEA-045的RUN-003已证明类名残差有效；本假设来自该模型的统一beta结构限制与第一性原理分析。
base_commit: 994ad823c425ed237a2956c1aa7057202a1e9673
core_change: 冻结TG-VPR/TST/CCGR与NCRA基准，只训练一个语义分歧斜率；不使用人工属性。
success_condition: H大于NCRA父模型77.201125，U和S任一项下降不超过2个百分点，斜率不饱和。
failure_condition: H不提高，或最优斜率达到98%边界。
experiment: V2-INNOVATION-012
result: RUN-002达到H=77.290521，相对NCRA父模型提高0.089396；作为辅助改进成立但增益较小，不晋级论文核心创新。
