# V2-INNOVATION-047 结果

状态：`rejected_direction_no_gain`。

RUN-001最高严格退回SDCR父模型`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`，selected iteration=`-1`、gamma=`0`。近邻平均相似度=`0.770630`，基础原型与新增方向最大绝对余弦仅`6.89e-08`，证明构造边界正确。

训练期间gamma在正负方向振荡，所有非零条件均降低H。RESCUE-1只把Adam学习率从0.01降到0.001，其他公式、top-5、seed和训练量不变；若仍以gamma=0为best，则判定方向无效。

RUN-001模型SHA256：`229506ba5a1990c7a3ad1d364d9debfc0e6d5dc4d440f95271ce72858e9d6242`；最后checkpoint SHA256：`64c937a94e7eee9a06eb9e8201b1a1653f21200f5c43a1ebdac173deb0ebf8e5`。

RUN-002将学习率降到0.001后，gamma变化更平稳，但所有非零条件仍未超过父模型；best再次为`H=78.320510%`、selected iteration=`-1`、gamma=`0`。因此失败来自近邻正交差分方向本身，而非优化振荡，IDEA-081拒绝并提前止损。

RUN-002模型SHA256：`fe8bd9065565d287e0d55645dd68fe48862b829ad21f03675bc1658fdf527eea`；最后checkpoint SHA256：`37bf3b32df3813c13b1033a81b7a3b96cff302dfb109d4caf38f11d83c86078e`。
