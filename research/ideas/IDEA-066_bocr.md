# IDEA-066：Bi-Orthogonal Cross-LLM Residual

status: testing
problem: OCLR只去除了类名身份方向，Claude残差仍可能重复TG-VPR已经编码的GPT结构化类别原型。
hypothesis: 对Claude原型同时去除类名方向和当前TG-VPR父原型的Gram–Schmidt方向，可保留更纯的框架外语义并超过OCLR。
evidence_refs: IDEA-063 OCLR两seed可靠成立；IDEA-065证明正交化跨文本源有效；TG-VPR父原型是当前GPT结构化语义的直接代码事实源。
base_commit: 65bae6e19f2e054c6d0c7831ccc54be79ce557e2
core_change: 在OCLR类名正交化基础上，再对每类TG-VPR父原型的正交方向去投影；训练仍只学习beta。
success_condition: H大于OCLR最高78.072185，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过OCLR，或beta达到98%上限。
experiment: V2-INNOVATION-032
