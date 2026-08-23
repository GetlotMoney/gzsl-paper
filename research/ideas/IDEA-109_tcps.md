# IDEA-109：Triadic Competition Pair Selector

status: testing
problem: 现有pair selector只描述top1/top2关系；当top3也很接近时，二元纠错可能忽略第三类竞争而过度翻转。
hypothesis: 在稳定SNPS top-3图上加入parent top2-top3间隔，可区分孤立二元混淆与三类拥挤竞争，并进一步提高H。
evidence_refs: IDEA-106稳定top-3两seed结果；SDCR_ERROR_AUDIT_001细粒度同族多类竞争。
base_commit: 89db8894ffe3b756e090595517d946dcc6768543
core_change: 12维SNPS selector新增第13维top2_minus_top3_margin；关系图、loss与评估不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、best退回关闭态或第三类特征退化。
experiment: V2-INNOVATION-075
paper_core_innovation: false
