# IDEA-080：Regularized Multi-Geometry Sentence Routing

status: testing
rescue_of: IDEA-079
attempt_no: RESCUE-2
problem: MGSR有正H信号，但±0.25与±0.10两种输出上限都会饱和；故障来自共享系数被seen CE持续推大，而非单纯上限过宽。
hypothesis: 保留MGSR的±0.25表达空间，同时用0.05×mean(共享系数²)直接抑制系数增长，可得到非饱和类别路由并超过MGSR RUN-001。
evidence_refs: IDEA-079 RUN-001/002均在不同输出上限饱和；单纯收紧上限已失败。
base_commit: 5cedbf0a1996e49f61480461297568166395c74f
core_change: MGSR结构、父权重、seed和训练量不变，只新增0.05共享系数L2训练项。
success_condition: H大于78.365239，class variation大于0.001，最小权重大于0.01，且残差绝对最大值小于0.245。
failure_condition: H不超过MGSR RUN-001、残差继续饱和或类别差异退化。
experiment: V2-INNOVATION-046
interim_result: RUN-001的0.05 L2过强，best退回父模型且class variation=0；最后补救固定为0.005 L2。
