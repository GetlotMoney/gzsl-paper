# V2-TUNE-006 结果

状态：`phase1_completed_all_three_pass`。

CUB与AWA2的正式RUN只读取`train_features/train_labels/class-name/text-v2`，没有加载official test，也没有训练模型。执行提交统一为`7545624fa4fca57fc057855b8fc478e2912b212a`。

| 数据集 | 文本 | 对应类余弦 | hardest-negative margin | 正margin类占比 | V→T top-1 | T→V top-1 | 角色间余弦 |
|---|---|---:|---:|---:|---:|---:|---:|
| CUB | class-name | 0.325889 | 0.020682 | 80.000% | 80.000% | 80.667% | — |
| CUB | text-v1 | 0.319383 | 0.005913 | 67.333% | 67.333% | 68.000% | 0.625400 |
| CUB | text-v2 | 0.347254 | 0.024894 | 83.333% | 83.333% | 83.333% | 0.892531 |
| AWA2 | class-name | 0.284673 | 0.067358 | 100.000% | 100.000% | 100.000% | — |
| AWA2 | text-v1 | 0.294092 | 0.043836 | 100.000% | 100.000% | 100.000% | 0.726709 |
| AWA2 | text-v2 | 0.312016 | 0.081102 | 100.000% | 100.000% | 100.000% | 0.804149 |
| SUN | class-name | 0.306123 | 0.023702 | 84.031% | 84.031% | 81.705% | — |
| SUN | text-v1 | 0.294122 | 0.013615 | 79.690% | 79.690% | 70.388% | 0.689083 |
| SUN | text-v2 | 0.328921 | 0.034420 | 89.767% | 89.767% | 82.636% | 0.799588 |

结论：CUB、AWA2和SUN全部通过seen-only恢复门槛。CUB相对class-name的margin提高0.004212；AWA2提高0.013743；SUN提高0.010718，V→T提高5.736个百分点。所有决策均未使用official test。

三数据集phase1已经完成。下一步按预注册计划运行Pure CLIP、Mean8和M3 seed 7恢复实验；Chen-style结果必须继续披露test_used_for_selection=true。
