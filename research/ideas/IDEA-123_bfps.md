# IDEA-123：Bias-Free Pair Selector

status: testing
problem: SNPS seed5/7权重余弦0.9567、范数接近，但selector bias差异较大0.0232/0.0402，可能是跨seed不稳定来源。
hypothesis: 固定全局bias为0，只训练12维证据权重，可去除统一top1偏置并提高跨seed稳定性。
evidence_refs: SNPS_TOP3_SELECTOR_AUDIT：weight cosine=0.956653，bias seed5/7=0.023213/0.040247；稳定top-3结果。
base_commit: 400afe2a7f30d47b8540668e7fc8b4fd88e3853e
core_change: 稳定SNPS selector_bias从可训练标量改为固定0；其余不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-089
paper_core_innovation: false
