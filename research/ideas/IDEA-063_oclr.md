# IDEA-063：Orthogonal Cross-LLM Residual

status: supported
problem: CLRE有效但beta接近较大值；Claude文本与类名平均余弦0.884752，部分证据可能已被NCRA类名分支重复覆盖。
hypothesis: 对每类Claude原型去除其类名方向，只保留单位球面切空间中的独立语义残差，可减少重复身份证据并超过MLRE最高H。
evidence_refs: IDEA-058证明Claude残差有效；IDEA-061/062证明Claude与merge混合退化；NCRA已显式编码类名身份，需隔离Claude独立分量。
base_commit: 8452f2ad849942174162c520d83ee16bcfaa33d3
core_change: CLRE训练不变，仅把Claude原型替换为对类名方向正交化后的单位残差。
success_condition: H大于MLRE最高77.829140，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过MLRE，或beta达到98%上限。
experiment: V2-INNOVATION-029
result: U/S/H/ZS=77.094042/79.075468/78.072185/84.185731%，H比MLRE高约0.243046且beta非饱和；强候选成立，provenance/新颖性待补。
