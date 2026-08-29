# IDEA-049：Class-Conditioned Patch Evidence

status: supported
problem: LPSR把所有类别共享的top64 patch平均成单一向量，丢失“某个局部块支持哪个类别”的定位关系，导致beta方向错误。
hypothesis: 对每个类别分别用正交局部文本残差在576个patch中选择top8，再把top8平均相似度作为类别条件logit残差，可以恢复局部定位并提高H或ZS。
evidence_refs: IDEA-048的负beta与ZS下降直接定位了class-agnostic pooling故障；真实patch cache和前六句局部描述均已SHA绑定。
base_commit: d8f590d8d2e4a7c3f4ae9f2105637d99fdecfb6c
core_change: 将class-agnostic局部向量改为每类独立top8 patch证据；冻结SEBC父模型，只训练一个融合beta，不使用人工属性。
success_condition: 相对SEBC父模型H提高至少0.10个百分点或ZS提高至少0.20个百分点，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H/ZS均未过门槛，或beta达到98%上限。
experiment: V2-INNOVATION-015
result: top2的RUN-003达到U/S/H/ZS=76.119131/79.278153/77.666533/83.168101%，相对SEBC父模型H提高0.148151并通过门槛；作为创新候选保留，相关工作检索前不作原创claim。
