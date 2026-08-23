# IDEA-069：Orthogonal Role-Matched Residual

status: testing
problem: OCLR证明Claude去类名方向有效；需要用不同生成模型与不同句子组织验证该机制是否能继续提高，而不是只在Claude/merge全局原型上成立。
hypothesis: 对GPT-5.6 role-matched七句语义取均值并去除类名方向，可提供比GPT-5.5父语义更新、又不重复类名身份的残差并超过OCLR。
evidence_refs: IDEA-063 OCLR两seed可靠；IDEA-065 OMLR跨源复现；GPT-5.6 role-matched均值与GPT-5.5均值余弦0.880657，且cache SHA已绑定。
base_commit: 597c012d6c4545c58a1a50ad17c604a21bf1be5f
core_change: CLRE训练不变，输入改为GPT-5.6七句role-matched均值并执行类名正交化。
success_condition: H大于OCLR最高78.072185，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过OCLR，或beta达到98%上限。
experiment: V2-INNOVATION-035
