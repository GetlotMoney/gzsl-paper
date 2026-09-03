# V7-ABLATION-004 结果

状态：planned。去父重训消融：冻结 TG+GTD source，head 基于 Mean8 纯文本 base，S/V/I 五组条件（Full / S-off / V-off / I-off / V+I-off）各自从 seed7 重新初始化完整训练 28,228 updates，另加 B0 零训练 Mean8 基线。

- code parent：`35cefc52896c383e1ec75a3adc5f78d218d616a3`
- head base：Mean8 纯文本（`tg_vpr.base_prototypes()`），非 TG+GTD 迁移原型
- 训练信号：S ← 普通 seen-only CE；V/I ← 一阶 class-held-out outer CE（TUNE014 机制）；无方向 CE
- 目的：区分“TG+GTD 遮蔽 V/I”与“CLS+一文本语义路径已到天花板”
- 架构耦合（双Agent交叉审查确认）：Reader 的唯一 logits 通道是关系分支 `readout @ alpha*compiled_g`；I-off 关系置零后 Reader outer 梯度为零、保持 READER_SEED 初始化，I-off 与 V+I-off 逐值等价，Full−I-off = 联合 V+I 贡献（不能单独归因 I）

正式 RUN 未启动；本页在完成后回填各组 U/S/H/ZS、best_update、相对 Mean8 与 formal V7 差值，并自检 I-off 与 V+I-off 逐值一致。
