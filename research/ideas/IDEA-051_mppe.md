# IDEA-051：Multi-Part Patch Evidence

status: rejected
problem: CCPE先把六句局部描述平均为一个文本向量，只能定位一个混合部位；SCPE进一步要求top2相邻，反而与多部位描述冲突。
hypothesis: 保留六句局部文本，每句分别在576个patch中寻找自己的最佳匹配，再平均六个部位证据，可同时覆盖嘴、翅、尾等分散细节并超过CCPE。
evidence_refs: IDEA-049证明类别条件patch定位有效；IDEA-050证明多个有效局部证据不必空间相邻；TG-VPR缓存前六句物理顺序固定为局部描述。
base_commit: 3f24e61b292ef7d95fdc6131244b2709740908f4
core_change: 将单个平均局部文本改为六个独立部位文本，每句独立top1 patch后按类别平均；冻结SEBC父模型，只训练一个beta。
success_condition: H大于CCPE最高77.666533，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过CCPE，或beta达到98%上限。
experiment: V2-INNOVATION-017
result: 所有非零beta均降低H，best退回SEBC关闭态；六路最大匹配累积了常见羽色与背景噪声。
