# IDEA-071：Adaptive Orthogonal Sentence Routing

status: testing
problem: OESR八句固定等权平均，两seedH可靠但U/ZS低于OCLR；不同句子可能对seen/unseen迁移贡献不同。
hypothesis: 固定OESR beta，从严格等权起步，只学习8个全局softmax句子权重，可突出更可迁移的句子并提高H或恢复U/ZS。
evidence_refs: IDEA-070 OESR两seed最高H=78.105812；IDEA-001固定等权适合原TG-VPR三组，但不等于GPT-5.6八句残差也必须等权。
base_commit: 442b28a44d5c79cbe2aad73c31598b7ccbae7ee4
core_change: 固定OESR beta，将八句等权均值改为可训练全局softmax权重；零初始化严格复现OESR。
success_condition: H大于OESR最高78.105812，U和S任一项下降不超过2个百分点，权重std大于0.01且min大于0.01。
failure_condition: H不超过OESR，或权重塌缩。
experiment: V2-INNOVATION-037
