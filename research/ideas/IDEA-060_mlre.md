# IDEA-060：Merged-LLM Residual Evidence

status: supported
problem: CLRE证明独立Claude描述有效，但Claude与GPT全局原型平均余弦仍有0.868739；本地merge文本原型与GPT仅0.852414，可能提供更完整的跨LLM综合语义。
hypothesis: 在同一SEBC父模型上用merge文本原型替代Claude残差并只训练beta，可超过CLRE并进一步提高ZS/H。
evidence_refs: IDEA-058证明跨LLM全局残差有效；IDEA-059说明不要与patch硬叠加；merge cache与GPT/Claude均不同且SHA已绑定。
base_commit: d281f8b01b6ebeae58c922516bdc3b00a06d6656
core_change: 保持CLRE训练和公式不变，仅把残差文本输入改为merge原型；不使用人工属性。
success_condition: H大于CLRE最高77.808093，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过CLRE，或beta达到98%上限。
experiment: V2-INNOVATION-026
result: H=77.829140略高于CLRE 0.021047，但ZS更低；保留为弱H候选，不作全面改进或原创claim。
