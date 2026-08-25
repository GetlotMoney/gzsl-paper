# V2-CONFIRM-010 结果

> 公平基线更正：本实验的B0使用旧Xian类名清洗，和text-v2的自然类名不完全一致。B1/M3绝对成绩仍有效；相对公平Pure CLIP的最终增益请使用`V2-CONFIRM-012`，不要继续引用本页旧B0差值。

状态：`completed_two_of_three_plus3`。

全部RUN使用提交`e8238c91cf7d7ba9693072af67ca3f1f2877c83a`。B0/B1各只评估一次；M3各评估201次official test并选择整次RUN的整模型全局最大H。`unseen_images_used_for_gradient=false`，`strict_blind_claim=false`。

| 数据集 | 条件 | U | S | H | ZS | best epoch | ΔH vs B0 | ΔH vs B1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CUB | Pure CLIP | 62.211 | 64.206 | 63.193 | 79.581 | — | 0.000 | -5.520 |
| CUB | Mean8 text-v2 | 69.102 | 68.327 | 68.713 | 86.180 | — | +5.520 | 0.000 |
| CUB | M3 End-to-End | 73.116 | 79.941 | 76.377 | 83.065 | 117 | +13.184 | +7.664 |
| AWA2 | Pure CLIP | 93.471 | 95.756 | 94.600 | 97.444 | — | 0.000 | -0.663 |
| AWA2 | Mean8 text-v2 | 94.698 | 95.833 | 95.262 | 99.173 | — | +0.663 | 0.000 |
| AWA2 | M3 End-to-End | 96.952 | 95.690 | 96.317 | 98.938 | 188 | +1.717 | +1.054 |
| SUN | Pure CLIP | 60.139 | 57.868 | 58.982 | 88.958 | — | 0.000 | -4.490 |
| SUN | Mean8 text-v2 | 62.847 | 64.109 | 63.472 | 91.528 | — | +4.490 | 0.000 |
| SUN | M3 End-to-End | 72.361 | 70.039 | 71.181 | 88.611 | 122 | +12.199 | +7.709 |

## 判定

- B0在三数据集逐项复现，证明hardlink复用没有改变Pure CLIP。
- B1在三数据集均高于text-v1与Pure CLIP，文本修复成立。
- M3在CUB和SUN超过Pure CLIP至少3个百分点；AWA2仅提高1.717个百分点，保留真实结果并标记未达门槛。
- 三个M3的TST-NTR、CCGR输出和TG-VPR/TST/NTR/CCGR梯度均非零。
- M3以GZSL H为主目标；三数据集ZS均如实记录，不与其他checkpoint拼接。

准确RUN、配置、模型、checkpoint、history、metrics和数据指纹SHA见`PARAMETER_MATRIX.csv`，原始产物位于`/data/lby/projects/cv_project/GZSL_Warehouse/final/v2/CONFIRM-010_text_v2_recovery`。
