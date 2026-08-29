# IDEA-070：Orthogonal Eight-Sentence Residual

status: supported
problem: GPT-5.6 role-matched七句正交残差为正但低于OCLR；另一份GPT-5.6八句cache采用不同生成组织，可能保留更多全局细节。
hypothesis: 对GPT-5.6八句语义取均值并去除类名方向，可超过role-matched七句版本并挑战OCLR。
evidence_refs: IDEA-069 ORMR H=77.935371；IDEA-063 OCLR两seed可靠；GPT-5.6八句均值与GPT-5.5均值余弦0.885720，cache SHA已绑定。
base_commit: 70bcdfec2d308f33c64f7928516d7a87dbe94e8c
core_change: ORMR训练和正交公式不变，仅将语义来源改为GPT-5.6八句均值。
success_condition: H大于OCLR最高78.072185，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过OCLR，或beta达到98%上限。
experiment: V2-INNOVATION-036
result: seed5/7 H=78.105812/78.102514，差距仅0.003299且均超过OCLR；当前最高H取seed5，但U/ZS仍低于OCLR。
