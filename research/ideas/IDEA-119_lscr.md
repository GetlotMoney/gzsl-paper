# IDEA-119：Local Semantic Competition Resolver

status: testing
problem: 现有pair selector只在top1/top2之间修正，无法处理三个相近细粒度类别共同竞争。
hypothesis: 对相关top3同时聚合11种文本证据并学习零和三类修正，可解决多类局部竞争并提高H。
evidence_refs: SDCR_ERROR_AUDIT_001显示同族多类混淆；IDEA-106 top-3语义关系稳定；IDEA-109仅把top3当标量上下文无效。
base_commit: ad1687793e35359a049ba417ea9a6fb786fe7c75
core_change: 二元pair修正替换为相关top3三类零和修正，训练使用真实类包含的三类局部CE。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、top3训练标签退化或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-085
paper_core_innovation: false
