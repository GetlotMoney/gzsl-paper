# V7-ABLATION-004 结果

状态：completed（诊断性重训消融，`decision=diagnose_no_parent_ablation`）。

冻结 TG+GTD source、head 基于 Mean8 纯文本 base、S/V/I 五组条件从头重训 + B0 零训练 Mean8 基线。全部 RUN 使用冻结 commit `28331743292857aa0a225992c39694948026dac6`，seed7、28228 updates、141 eval、逐条件 best-H 选择、U/S/H 200 类竞争、ZS 50 unseen 类竞争、unseen 图像不产生梯度。训练信号：S ← 普通 seen-only CE；V/I ← 一阶 class-held-out outer CE（TUNE014 机制）；无方向 CE。

## 最佳结果（best-H 对应同一 checkpoint）

| condition | best_update | U | S | H | ZS | Δ vs Mean8 | Δ vs formal V7 |
|---|---|---|---|---|---|---|---|
| B0（Mean8 零训练） | 0 | 69.136261 | 68.369150 | 68.750566 | 86.146760 | — | -11.759866 |
| Full | 27213 | 76.349747 | 79.260629 | 77.777962 | 87.180197 | +9.027396 | -2.732469 |
| S-off | 9729 | 72.072846 | 76.055366 | 74.010570 | 82.854766 | +5.260003 | -6.499862 |
| V-off | 19176 | 78.680176 | 63.951403 | 70.555310 | 86.334091 | +1.804744 | -9.955122 |
| I-off | 13536 | 78.947043 | 63.304627 | 70.265792 | 86.192983 | +1.515226 | -10.244640 |
| V+I-off | 13536 | 78.947043 | 63.304627 | 70.265792 | 86.192983 | +1.515226 | -10.244640 |

重训模块贡献（Full − off）：**S-off +3.767 H；V-off +7.223 H；I-off +7.512 H（与 V+I-off 相等，即 V+I 联合贡献）**。

## 判断

- **TG+GTD 遮蔽假说强烈成立**：在纯 Mean8 文本基线上 head 有巨大可学内容（S +3.77H、V +7.22H、V+I 联合 +7.51H、总量 +9.03H），远大于完整框架内的边际贡献（ABL002：S +0.50H、V/I +0.25H）。说明 S/V/I 并非无效模块，而是 TG+GTD 把可判别信息编入迁移后原型、占满收益空间，V/I 在完整框架中已学不到新内容。
- **但 Full=77.78 仍低于 formal V7（80.51）2.73H、低于 TUNE013（79.95）2.17H** → TG+GTD 的迁移本身仍提供不可替代的约 2.7H 增益。
- 对方向的含义：本实验**不支持**“把旧 I 模块继续堆进完整框架”；它指向两条路——(a) 重构收益空间分配（让 TG+GTD 更轻、把判别容量让给 head）；(b) 找 TG+GTD 之外的**新学习信号 / 新表示原语 / 新视觉信息源**。弱基线上的涨幅不得当作完整框架下的模块独立增益（假 headroom 警告）。
- **架构耦合自检（实证）**：I-off 与 V+I-off 的 201 条评估轨迹逐值一致（仅 condition 标签与日志字段 `train_mean.vi_outer_ce` 不同），best_update 与 U/S/H/ZS 全同 → 证实 Reader 唯一 logits 通道是关系分支、I-off 中 Reader 梯度为零并保持 READER_SEED 初始化。

## 产物

- output：`/data/lby/projects/cv_project/GZSL_Warehouse/ablation/v7/V7-ABLATION-004_NO_PARENT_HEAD/{BASELINE-B0,RUN-FULL,RUN-S-OFF,RUN-V-OFF,RUN-I-OFF,RUN-VI-OFF}`
- RUN-S-OFF、RUN-I-OFF 使用 cuda:1 设备变体 config（仅 device 字段不同），其执行 config SHA 记录在 PARAMETER_MATRIX.csv；其余使用仓库内 config。
- 各组 metrics/model/history SHA 见 PARAMETER_MATRIX.csv。
