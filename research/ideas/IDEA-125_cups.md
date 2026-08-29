# IDEA-125：Class-Uniform Pair Selector

status: rejected
problem: SNPS pair训练可能被出现次数较多的seen类主导；此前平衡的是top1/top2标签而非真实类别。
hypothesis: 按真实seen类别pair频次做均值1逆频率权重，可使150类贡献更均匀并提高H。
evidence_refs: B-GWPS/M-BGWPS标签权重过强失败；MHPS样本缩减失败；稳定SNPS使用全量pair。
base_commit: 94fc604202d431e5743f60c41d23146bb9066cd1
core_change: 全量SNPS pair loss乘以真实类别逆pair频率均衡权重；标签分布、特征和推理不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710且最大类别权重温和；正提升后追加seed7。
failure_condition: H不超过top-3、类别权重退化或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-091
paper_core_innovation: false
result: H=78.462899，低于稳定top-3；真实类别权重最大33.268658并非温和平衡。
decision: 稀有类别pair过少导致极端逆频率权重，拒绝。
