# V2-TUNE-005 结果

状态：`completed_text_v1_diagnostic`。

三组RUN均只读取`train_features/train_labels/class-name/role-text`，没有加载official test，也没有训练模型。执行提交统一为`7f717f5e8dec782b1c41adfcef45ce5796a29980`。

| 数据集 | 文本 | 对应类余弦 | hardest-negative margin | 正margin类占比 | V→T top-1 | T→V top-1 | 角色间余弦 |
|---|---|---:|---:|---:|---:|---:|---:|
| CUB | class-name | 0.325889 | 0.020682 | 80.000% | 80.000% | 80.667% | — |
| CUB | text-v1 | 0.319383 | 0.005913 | 67.333% | 67.333% | 68.000% | 0.625400 |
| CUB | old-eight-role（仅诊断） | 0.346107 | 0.024797 | 84.000% | 84.000% | 80.667% | 0.842785 |
| AWA2 | class-name | 0.284673 | 0.067358 | 100.000% | 100.000% | 100.000% | — |
| AWA2 | text-v1 | 0.294092 | 0.043836 | 100.000% | 100.000% | 100.000% | 0.726709 |
| SUN | class-name | 0.306123 | 0.023702 | 84.031% | 84.031% | 81.705% | — |
| SUN | text-v1 | 0.294122 | 0.013615 | 79.690% | 79.690% | 70.388% | 0.689083 |

结论：CUB和SUN的text-v1明显降低seen类边界；AWA2对应类余弦虽略高，但最难负类margin显著低于class-name。三套text-v1均不冻结为论文文本资产，继续生成并用相同seen-only指标筛选text-v2。CUB旧八角色证明新视觉缓存本身不是主要原因，但其编码元数据不完整，因此只保留为诊断证据。

完整产物位于`/data/lby/projects/cv_project/GZSL_Warehouse/diagnostics/v2/TUNE-005_seen_only_text_assets`；准确产物SHA记录在`PARAMETER_MATRIX.csv`及各RUN的`data_fingerprints.json`。
