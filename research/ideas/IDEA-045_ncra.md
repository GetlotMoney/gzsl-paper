# IDEA-045：Name-Conditioned Residual Alignment

status: testing
problem: GPT八角色长描述可能弱化直接类别身份，当前最佳无专家H=76.006848。
hypothesis: 标准CLIP类名原型提供独立身份语义，以零初始化有界logit残差融合可提高H和ZS。
base_commit: fab936a65fe0e8d066909bf82752517f44a41f8e
core_change: 冻结父模型，只训练一个类名CLIP残差beta；不使用人工属性。
success_condition: H大于76.006848；目标H大于等于77.023182。beta饱和只触发扩大已验证量纲范围，不单独否定方法。
failure_condition: 扩大到beta上限20后H仍不提高，或最优值仍处于边界且继续扩大不再带来收益。
experiment: V2-INNOVATION-011
