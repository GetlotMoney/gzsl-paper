# IDEA-062：Class-Adaptive Cross-LLM Mixture

status: rejected
problem: ACLM全局权重退化到Claude端点，但MLRE在H上略优，说明不同类别可能偏好不同LLM语义而全局比例无法表达。
hypothesis: 以每类Claude/merge余弦一致度为条件，用共享bias+slope预测200个类别混合权重，可避免全局端点退化并超过MLRE。
evidence_refs: IDEA-061全局权重退化；IDEA-058/060两个端点分别在ZS/H上占优；类别间Claude/merge一致度存在差异。
base_commit: 5928b6976d50d435007ec2683c911620763c1696
core_change: 将单一Claude权重改为由类别语义一致度驱动的200类权重，只训练共享bias和slope。
success_condition: H大于MLRE最高77.829140，U和S任一项下降不超过2个百分点，权重std大于0.01且mean位于(0.02,0.98)。
failure_condition: H不超过MLRE、权重近常数或退化端点。
experiment: V2-INNOVATION-028
result: H=77.811876低于MLRE；权重mean=0.990835/std=0.000207，退化为Claude常数端点。
