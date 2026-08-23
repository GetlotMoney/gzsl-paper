# IDEA-108：Reciprocity-Weighted Semantic Neighborhood Pair Selector

status: testing
problem: union top-5峰值最高但跨seed增量不一致；mutual top-5稳定却删除大量可能有效的单向边；union top-3稳定但有硬截断。
hypothesis: 对互为top-5边赋权1、单向top-5边赋权0.5，并同时缩放训练loss与推理修正，可在不硬截断的情况下兼顾覆盖与稳定性。
evidence_refs: IDEA-106 top-5/top-3结果；IDEA-107 mutual top-5两seed稳定结果。
base_commit: f96bea0bf7211dc09fba5cdb53b63eacadefe77a
core_change: 二值语义邻接改为互惠置信度{0,0.5,1}，置信度同时作用于pair训练权重和推理delta。
success_condition: seed5 H大于稳定top-3 78.466710或最高top-5 78.480710；否则拒绝，不做第4次图补救。
failure_condition: H不超过top-3、U/S任一下降超过2个百分点或置信度退化。
experiment: V2-INNOVATION-074
paper_core_innovation: false
interim_result: seed5 patch-free H=78.476148，相对稳定top-3 +0.009438、距最高top-5仅-0.004562；追加seed7。
