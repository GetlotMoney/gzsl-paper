# IDEA-093：Consensus-Contrarian Tie-Breaker

status: rejected
problem: AGCT两seed都学习负beta，表明Claude不是正向教师，而是对低margin共识错误提供反向诊断；AGCT在Claude与SDCR分歧时也触发，可能浪费校正。
hypothesis: 只在Claude与SDCR top1一致且同族低margin时启用AGCT，负beta会反转共享偏好、集中纠正共识错误，并超过AGCT。
evidence_refs: IDEA-092两seed均学到约-2 beta并小幅提高H；IDEA-085证明Claude全局正向叠加无效。
base_commit: 1a7294ab515970cc20f6a4aeef2530cf6dc2a1ad
core_change: AGCT公式和train-only阈值不变，只新增Claude必须与SDCR top1一致的门控条件。
success_condition: H大于78.339523，U和S任一项下降不超过2个百分点，gate非零且beta不饱和。
failure_condition: H不超过AGCT、gate过稀或反共识规则伤害正确共识。
experiment: V2-INNOVATION-059
result: 共识条件把gate降至约0.09，beta升到正4.57仍不改变任何official指标；反共识假设拒绝。
