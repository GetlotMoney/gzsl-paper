# IDEA-096：Gated Pair Evidence Selector

status: testing
problem: AGCT gate包含大量可纠正top2，但任何固定来源/方向都会同时纠正和破坏；需要按样本学习top1/top2选择。
hypothesis: 用parent margin及Claude、merge、patch三个证据差值训练跨类别共享的四维线性pair selector，可从seen同族pair学习可迁移规则并超过AGCT。
evidence_refs: AGCT_SOURCE_ORACLE_001；IDEA-092固定25分位两seedsupported；IDEA-094/095关闭固定多源与patch方向。
base_commit: cae6eedcaf2d6ccba81bb402d9f1c0cdb00c0c79
core_change: 固定SDCR与25分位gate；仅在train seen真类位于top2的样本上，用四维共享选择器和成对CE训练5个参数。
success_condition: H大于78.357224，U和S任一项下降不超过2个百分点，pair训练集两标签均存在且selector参数有限。
failure_condition: H不超过AGCT、selector退回零、seen pair过拟合或跨域选择方向相反。
experiment: V2-INNOVATION-062
