# IDEA-055：Episodic Class-Conditioned Patch Evidence

status: testing
problem: CCPE beta由普通seen CE训练；它能提高H，但训练目标没有显式模拟seen→unseen的局部证据迁移。
hypothesis: 保持CCPE top2公式不变，在三个100/50类class-exclusive episode中用50类pseudo-unseen图像训练beta，可学到更适合真实unseen竞争的局部证据强度并超过CCPE。
evidence_refs: IDEA-049证明CCPE公式有效；IDEA-047证明class-exclusive episode可成功学习可迁移的竞争参数；IDEA-054说明继续增加同源分数统计无效，应改变训练目标。
base_commit: 476c92576ed3d69f30469fe06e0974ad1286d6aa
core_change: CCPE公式与输入不变，将beta训练从普通seen CE改为三个100/50类balanced episode；真实unseen图像不进入梯度。
success_condition: H大于CCPE最高77.666533，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过CCPE，或beta达到98%上限。
experiment: V2-INNOVATION-021
