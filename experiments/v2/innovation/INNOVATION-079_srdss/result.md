# V2-INNOVATION-079 结果

状态：`rejected_best_parent`。

RUN-001初始态逐项复现SNPS top-3：`U/S/H/ZS=76.883179/80.116844/78.466710/84.121209%`。模型仅有一个可训练`scale_weight`，完整执行28,228次更新后所有非零状态均低于父模型，selected iteration=`-1`、best scale weight=`0`。

分阶段训练边界有效，但尺度系数不能独立叠加；RDSS seed5高峰依赖13维联合协调。IDEA-113拒绝，不追加seed7。

模型SHA256：`3c7f7bcc02462437c6430827c14fddc07a41618be263f0473918989355114649`；最后checkpoint SHA256：`b36d22ce79842dd840e8134bebc612ed0e85dfc748e520a125362931f6a62229`。
