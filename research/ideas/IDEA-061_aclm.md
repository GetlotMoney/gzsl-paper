# IDEA-061：Adaptive Cross-LLM Mixture

status: rejected
problem: CLRE具有更高ZS，MLRE具有略高H；两个端点各有优势，但直接二选一无法兼顾。
hypothesis: 固定CLRE与MLRE各自已训练beta，只学习Claude/merge残差之间的凸混合比例，可在两个端点之间找到更好的H-ZS折中并超过MLRE。
evidence_refs: IDEA-058的CLRE H=77.808093/ZS=83.523118；IDEA-060的MLRE H=77.829140/ZS=83.225495；二者共享SEBC父模型且仅文本输入不同。
base_commit: d8561d3008754628ac01d89ace2fd536c4ba36a1
core_change: 固定CLRE/MLRE beta，在两个缩放后的残差logits之间只训练一个全局凸混合权重。
success_condition: H大于MLRE最高77.829140，U和S任一项下降不超过2个百分点，混合权重位于(0.02,0.98)。
failure_condition: H不超过MLRE，或权重退化到端点。
experiment: V2-INNOVATION-027
result: 最高H=77.811876低于MLRE，Claude权重0.980989退化到端点；全局混合失败。
