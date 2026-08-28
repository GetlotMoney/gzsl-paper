# V3-CONFIRM-014 结果

状态：`completed`。

本Experiment回答的控制性问题是：在四个RUN都从seed7 fresh TG开始、主batch与TG学习率时间线一致、禁止加载任何CUB训练checkpoint时，GTD、MMT、BD是否仍有独立效果。

| RUN | 条件 | U | S | H | ZS | 相对TRY042 ΔH | 同checkpoint移除 ΔH | `|U-S|` | 决策 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| V3-TRY-042 | TG | 77.309537 | 75.999695 | 76.649020 | 86.146760 | — | — | 1.309842 | 匹配控制 |
| V3-TRY-043 | TG+GTD | 79.758894 | 76.615125 | 78.155408 | 85.828280 | +1.506388 | +1.506388 | 3.143770 | 通过，第一创新候选 |
| V3-TRY-044 | TG+MMT | 78.689486 | 76.825291 | 77.746215 | 86.917830 | +1.097195 | +1.379207 | 1.864195 | 通过，但只作GTD同类替代设计 |
| V3-TRY-045 | TG+BD | 71.382040 | 78.531164 | 74.786137 | 85.436910 | -1.862884 | +18.821217 | 7.149124 | 失败；相对匹配父条件退化 |

GTD与MMT都通过预注册双门和U/S差门槛，但两者都在Mean8→Value语义方向移动unseen原型，并由Gate控制移动幅度，因此是同一瓶颈上的竞争方案，不能包装成两个核心创新。按主指标H选择GTD作为第一创新候选；MMT保留为方法选择消融；BD淘汰，不继续调参或在其代码上堆叠。

四条正式RUN均绑定`13bd1ccb513710ce798fbaa7147af447d43b0b36`，满足`loaded_training_checkpoints=[]`、`stop_reason=completed_fixed_150`、`total_updates=21171`和`history_length=152`。每条RUN的U/S/H/ZS及Full/Off来自各自同一个best-H checkpoint。完整测试为`547 passed, 1 skipped, 3 subtests passed`；两轮审核均无P0/P1。当前只有seed7，结果足以筛选候选，不能声称跨seed稳定性。
