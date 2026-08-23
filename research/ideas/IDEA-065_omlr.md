# IDEA-065：Orthogonal Merged-LLM Residual

status: testing
problem: OCLR正交化Claude后显著提高U/H/ZS；需要验证去类名重复方向是否是跨文本源原则，而不是Claude特例。
hypothesis: 对merge文本原型执行同样的类名正交化并训练beta，可超过OCLR并保留MLRE较高S的优势。
evidence_refs: IDEA-063 OCLR strong candidate；IDEA-060 MLRE为弱H候选；两者唯一关键差异是文本源及是否正交化。
base_commit: 02fdd7ded27b13993fa27ce5c7f4fedc07abc5a4
core_change: MLRE训练不变，仅将merge原型对类名方向正交化。
success_condition: H大于OCLR最高78.072185，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过OCLR，或beta达到98%上限。
experiment: V2-INNOVATION-031
