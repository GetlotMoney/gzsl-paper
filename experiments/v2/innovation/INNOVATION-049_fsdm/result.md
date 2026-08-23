# V2-INNOVATION-049 结果

状态：`rejected_seen_metric_bias`。

RUN-001初始化完整复现SDCR，但所有非单位度量条件都降低H，训练中后段长期约为`77.8–78.0%`；best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`、selected iteration=`-1`、权重全1。

RSDM单分支与FSDM三分支均失败，说明当前seen CE学习CLIP维度度量会产生跨类域偏置，而不是度量放置范围不够。IDEA-083拒绝并关闭当前无专家链的共享对角度量方向。

模型SHA256：`08426dd5d6bdca67e070f667eadec87e03d4e62993d1f6746c8c3cd77727230e`；最后checkpoint SHA256：`38f0e9df79a5151f374826d24ddc6a01610b29208f82bdbd40ef4d228775d0d9`。
