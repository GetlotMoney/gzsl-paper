# IDEA-013：SGT语义图残差迁移

```yaml
idea_id: IDEA-013
source_type: first_principles_relational_transfer
evidence_refs: [V2-INNOVATION-002, V2-TRY-040]
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TST逐类独立决定迁移，无法利用相似鸟类之间共享的视觉残差结构。
hypothesis: 从pseudo-seen或seen类别真实学到的切空间残差，经文本KNN图传播给pseudo-unseen或true-unseen，可提供独立于Value路径的关系迁移信号。
core_change: 在TST原型上增加文本top-5图传播的seen残差，并用seen类episode训练有界传播强度；图边不使用unseen图像。
success_condition: seed7相对TG-VPR+TST最高H提高至少0.20个百分点，U和S各自下降不超过2个百分点，传播强度不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: rejected
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: none
last_attempt: V2-TRY-045
last_decision: drop
```

SGT训练只使用150个seen类构造图迁移episode；true-unseen阶段只读取文本原型建立边，official图像在训练结束后才加载。传播强度为0时严格回到TST。

## V2-TRY-044结果

纯文本top-5图使`S`提高`0.475156`，但`U`下降`0.849992`、`ZS`下降`0.471133`，相对TST `Delta H=-0.242770`。补救1在边权中加入源视觉残差与目标TST切向方向的一致性，抑制视觉迁移方向相反的文本近邻。

## V2-TRY-045结果与止损

方向一致边仍使`U`下降`0.871569`、`ZS`下降`0.802213`，相对TST `Delta H=-0.222529`。两种边权均显示seen视觉残差不能通过文本近邻可靠迁移，IDEA-013提前止损并标记`rejected`。
