# IDEA-101：Nonlinear Pair Selector

status: rejected
problem: GWPS线性selector两seed可靠，但四种证据对不同margin区间可能存在非线性交互；pair数据轴与平衡轴均已收口。
hypothesis: 零输出初始化的4→8→1小型MLP可学习证据交互，在保持GWPS数据与推理边界的同时超过78.414246。
evidence_refs: IDEA-097两seedsupported；IDEA-098/099/100关闭pair权重与范围轴。
base_commit: 47ddadb73e4f21c80b18624301f03478bc46be88
core_change: GWPS其他部分全部固定；线性selector替换为hidden=8的GELU MLP，末层零初始化。
success_condition: H大于78.414246，参数有限、末层不塌缩且U/S任一下降不超过2个百分点。
failure_condition: H不超过GWPS、MLP过拟合或末层退回零。
experiment: V2-INNOVATION-067
result: MLP正常更新且H=78.414029，但低于线性GWPS 0.000217；复杂化无收益。
