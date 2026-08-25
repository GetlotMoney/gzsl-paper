# V2-TUNE-006 结果

状态：`phase1_sun_prerun_cub_awa2_completed`。

CUB与AWA2的正式RUN只读取`train_features/train_labels/class-name/text-v2`，没有加载official test，也没有训练模型。执行提交统一为`7545624fa4fca57fc057855b8fc478e2912b212a`。

| 数据集 | 文本 | 对应类余弦 | hardest-negative margin | 正margin类占比 | V→T top-1 | T→V top-1 | 角色间余弦 |
|---|---|---:|---:|---:|---:|---:|---:|
| CUB | class-name | 0.325889 | 0.020682 | 80.000% | 80.000% | 80.667% | — |
| CUB | text-v1 | 0.319383 | 0.005913 | 67.333% | 67.333% | 68.000% | 0.625400 |
| CUB | text-v2 | 0.347254 | 0.024894 | 83.333% | 83.333% | 83.333% | 0.892531 |
| AWA2 | class-name | 0.284673 | 0.067358 | 100.000% | 100.000% | 100.000% | — |
| AWA2 | text-v1 | 0.294092 | 0.043836 | 100.000% | 100.000% | 100.000% | 0.726709 |
| AWA2 | text-v2 | 0.312016 | 0.081102 | 100.000% | 100.000% | 100.000% | 0.804149 |

结论：CUB与AWA2均通过seen-only恢复门槛。CUB相对class-name的margin提高0.004212，双向检索分别提高3.333和2.667个百分点；AWA2 margin提高0.013743且双向top-1保持100%。这两项决策没有使用official test。

SUN文本已完成止损修正并生成内容寻址资产`bfe12cda3c37abdb`；其seen-only RUN已预注册但尚未运行。三数据集phase1全部完成前，不启动B0/B1/M3的Chen-style恢复训练。
