# V2-CONFIRM-012 结果

状态：`completed`。

本实验只重新计算三数据集Pure CLIP B0。图像缓存、标签和八角色文本embedding与`V2-CONFIRM-010`逐文件复用；唯一变量是使用冻结text-v2共同前缀重新编码的`class_name_embeds.pt`。B1与M3不重训，完成后只重新计算其相对公平B0的差值。

首次启动的CUB与AWA2配置使用了运行器未注册的condition ID，实际回退为Mean8原型；两项产物保留并标记`failed_runtime`，不作为B0结果。修正后的运行器会在此类歧义发生时直接拒绝配置，正式重跑使用全新输出目录。

三个正式B0均使用提交`2fc5312647b55e13e36e2b30d16479d713c8bbe4`，无训练、无checkpoint选择，official test只评估一次。

| 数据集 | 条件 | U | S | H | ZS | ΔH vs canonical B0 |
|---|---|---:|---:|---:|---:|---:|
| CUB | canonical Pure CLIP B0 | 66.659 | 64.136 | 65.373 | 83.268 | 0.000 |
| CUB | existing Mean8 B1 | 69.102 | 68.327 | 68.713 | 86.180 | +3.339 |
| CUB | existing M3 | 73.116 | 79.941 | 76.377 | 83.065 | +11.003 |
| AWA2 | canonical Pure CLIP B0 | 89.462 | 96.870 | 93.019 | 97.487 | 0.000 |
| AWA2 | existing Mean8 B1 | 94.698 | 95.833 | 95.262 | 99.173 | +2.244 |
| AWA2 | existing M3 | 96.952 | 95.690 | 96.317 | 98.938 | +3.298 |
| SUN | canonical Pure CLIP B0 | 60.208 | 60.736 | 60.471 | 89.653 | 0.000 |
| SUN | existing Mean8 B1 | 62.847 | 64.109 | 63.472 | 91.528 | +3.000 |
| SUN | existing M3 | 72.361 | 70.039 | 71.181 | 88.611 | +10.710 |

结论：规范化类名后，Mean8在三个数据集都高于Pure CLIP；现有M3相对公平B0分别提高`11.003 / 3.298 / 10.710 H`，三数据集均通过至少`+3 H`门槛。B1/M3绝对成绩来自`V2-CONFIRM-010`，没有重训或改写。
