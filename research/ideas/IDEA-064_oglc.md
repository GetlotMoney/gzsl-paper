# IDEA-064：Orthogonal Global-Local Composition

status: testing
problem: 原始CLRE与CCPE组合失败，但OCLR已去除类名重复方向并显著提高U/H/ZS，可能与GPT局部patch证据更独立。
hypothesis: 固定OCLR与CCPE两个best，只协调CCPE局部分支±25%，可在正交全局语义上恢复局部细节增益并超过OCLR。
evidence_refs: IDEA-063 OCLR H=78.072185/ZS=84.185731；IDEA-049 CCPE局部证据成立；IDEA-059原始Claude组合失败可能由身份重复导致。
base_commit: c1b4f164abb6a230099c9dde51ac2d942eba16a2
core_change: 将CLEC的CLRE全局父分支替换为OCLR正交全局分支，固定两套beta，只训练局部比例。
success_condition: H大于OCLR最高78.072185，U和S任一项下降不超过2个百分点，patch scale不在0.75/1.25边界。
failure_condition: H不超过OCLR，或scale到达边界。
experiment: V2-INNOVATION-030
