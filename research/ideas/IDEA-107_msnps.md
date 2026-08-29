# IDEA-107：Mutual Semantic Neighborhood Pair Selector

status: supported
problem: SNPS的union top-5约800条边，增量在seed5/7一正一负，可能含单向相似噪声。
hypothesis: 只保留双方互为top-5的稳定语义邻居，可维持seed5收益并修复跨seed不一致。
evidence_refs: IDEA-106最高H=78.480710但相对C-RGWPS增量跨seed不一致；SNPS语义图边数约800。
base_commit: 2e78bae9b101761b2c0ad7854bc54d6ec13c6983
core_change: SNPS语义邻接从union top-5改为mutual top-5；其余公式、参数和评估不变。
success_condition: seed5 H大于同seed C-RGWPS 78.393178，且若追加seed7则相对C-RGWPS两seed同号。
failure_condition: H不超过C-RGWPS或仍出现跨seed增量方向不一致。
experiment: V2-INNOVATION-073
paper_core_innovation: false
result: seed5/seed7 patch-free H=78.459247/78.426898，相对同seed C-RGWPS +0.066069/+0.014189，两seed同号且自身范围0.032349。
decision: mutual top-5修复了SNPS增量跨seed不一致，保留为稳定辅助候选；相关工作检索前不作核心原创claim。
