# IDEA-053：Dual-Scale Patch Evidence

status: rejected
problem: CCPE绝对top2证据达到H=77.666533；CNPE相对seen参考证据虽低于CCPE，但U和S同时提高，说明两者捕获的信息并不相同。
hypothesis: 联合训练绝对top2权重和seen参考z-score权重，可同时利用原始局部匹配强度与类别异常度，超过两个单独分支。
evidence_refs: IDEA-049提供绝对top2正证据；IDEA-052提供独立归一化正证据，且其beta非饱和、U/S同向改善。
base_commit: 3462a34b2333dce828f12acb8c3303ee99cfa810
core_change: 同一top2 patch分数分成绝对值与seen参考z-score两路，各训练一个有界beta；冻结SEBC父模型，不使用人工属性。
success_condition: H大于CCPE最高77.666533，U和S任一项下降不超过2个百分点，两个beta均不饱和。
failure_condition: H不超过CCPE，或任一beta达到98%上限。
experiment: V2-INNOVATION-019
result: 联合训练最高H=77.565132；分阶段训练所有非零归一化beta均低于CCPE，best退回absolute=9.177013/normalized=0，双尺度不互补。
