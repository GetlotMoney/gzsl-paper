# V2-ABLATION-004 结果

状态：`completed_mixed_seed7`。

三条Stagewise RUN均使用提交`04b3506b508902b0898c267b498ba8dcd0fda134`，固定50/100/50边界，整次RUN只选择一个全局best-H；阶段边界不根据test变化。

| 数据集 | End-to-End H | Stagewise H | ΔH（Stagewise-E2E） | Stagewise best epoch | best stage | 判定 |
|---|---:|---:|---:|---:|---|---|
| CUB | 76.376766 | 77.458777 | +1.082010 | 54 | TRANSFER_CCGR | Stagewise更高 |
| AWA2 | 96.316627 | 95.666877 | -0.649750 | 45 | TG_ONLY | Stagewise更低，且best不含TST-NTR/CCGR |
| SUN | 71.180998 | 68.329691 | -2.851307 | 189 | JOINT_FINETUNE | Stagewise更低 |

## 判定

- End-to-End在AWA2和SUN胜出，但在CUB低于Stagewise 1.082，未满足“第三个数据集不得低超过0.1”的全局切换门槛。
- seed7只是策略初筛，不从单seed宣布全局策略；后续三seed表必须同时保留两种策略。
- AWA2 Stagewise的全局best位于TG-only，因此不能当作完整三模块最佳模型。
- 三条RUN的阶段梯度边界均正确；CUB和SUN被选checkpoint的迁移与CCGR输出非零。

准确模型、checkpoint、history、metrics与数据指纹SHA见`PARAMETER_MATRIX.csv`。
