# IDEA-220 / Reliability-Gated Dual-Coordinate Fusion (RG-DCF)

- status: `rejected_at_prequeue_gate`
- problem_category: `reliability_robustness`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-219 / PADC`
- performance_status: `below_parent`

## 路径与模块

RG-DCF把稳定的mean8全局语义坐标S和允许旋转prototype副本的局部视觉坐标V分离，再用3-fold类别留出的seen-train OOF预测学习可靠性I。I只读取S/V的margin、entropy、top1互排、agreement与分布相关性，以hard abstaining switch决定保留S或切到V；不使用类别one-hot、教师、专家属性、官方test拟合或unseen梯度。

- S输入/输出：全局CLIP图像特征＋8句角色文本原型 -> 冻结200-way语义logits。
- V输入/输出：36patch＋V私有mean8副本 -> shared rank16投影、空间分离Top2 -> 200-way局部视觉logits。
- I输入/输出：S/V分布统计 -> 可靠性概率 -> S或V的最终决策。
- old_solution_path: 默认S/V同坐标并对全类别直接相加。
- new_solution_path: 双坐标专家 -> OOF正确性分歧监督 -> 可拒绝的条件选择。
- principle_difference: 学习对象从共同相似度变为专家可靠性状态。
- non_equivalence_test: Full必须胜S、全局混合、温度融合、entropy/margin规则和shuffled reliability。
- minimal_viability: OOF AUC、真实模块移除、coverage、damage/recovery与官方H/ZS共同过门。
- current_advantage: OOF可靠性信号强，但无法转化为1点H增益，真实Gate失败。
- failure_boundary: 全空间V太弱，即使可靠性排序有效，安全阈值仍会退化为几乎总选S。
- why_not_module: Full未产生非平凡部署行为，不形成成立的交互创新。
- paper_level_claim: none。

## 审核身份

- 最终Idea草稿SHA：`df01eee4abc380dec7c4bcde84221e8972115f88da135f1e8c6db6e3ad502d27`。
- 最终Idea A/B审核SHA：`f054ad8aaef51a87099f59c4c8ace2ed256636590c96036edec7825ecf6c398c` / `5648aaefea4a681f7ed4db03c7de1fa11124a6d736a092e8c6773c5d993f922e`，均为P0/P1/P2=0并通过。
- Gate脚本SHA：`95bdeb8aafdc451178fa59b4c8b5af02e6b0f06475dd8244326d5acf827f16e9`。
- 代码交叉复核A/B报告SHA：`896fdcc810d27f3e8fee4b9ac7b219186715e516164df458737dc7c0870d2483` / `4480dd08b1b33b3db45a6efecd237974f31f280c182a7101fbc64f227371e4d8`；双方P0/P1=0并通过。

## 真实结果

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/rgdcf/IDEA-220-GATE0/result.json@sha256:3cc2658c6ca0c86f77a5878c857289faf24022abf6b6575d181f942b9c4874f5`；日志SHA：`1f99277037fb98fc5ddbe57b4839e39707b26a6bfad0c7fd349ea9dbbd779476`。

| 条件 | U | S | H | ZS |
|---|---:|---:|---:|---:|
| class-name | 62.2104 | 64.2058 | 63.1923 | 79.6813 |
| mean8 S | 69.1363 | 68.3691 | 68.7506 | 86.1468 |
| shared-coordinate V | 33.0541 | 51.7246 | 40.3335 | 58.6731 |
| I-off全局混合 | 69.1363 | 68.3691 | 68.7506 | 86.1468 |
| Full RG-DCF | 69.1696 | 68.3691 | 68.7670 | 86.1468 |
| S/V oracle | 73.6333 | 76.0063 | 74.8010 | 89.6730 |

OOF true-class-balanced AUC为`0.8671`，class-bootstrap下界`0.8365`，所以可靠性特征不是随机信号。但damage constraint把阈值选到`q=0.95`；官方seen/unseen的V覆盖仅`0.227%/0.202%`，只在unseen恢复1张且不伤害任何样本。Full相对S只`+0.0165 H`，visual/interaction/full-control/coverage门均失败。

结论：可靠性可以排序，但全200类硬切换的动作粒度过大。Rescue2改成S先产生小候选集，V只在候选内验证，I只决定是否交换S top1，从根源上缩小一次错误动作的代价。

披露：`test_used_for_selection=true`、`test_used_for_hyperparameter_selection=true`、`nested_official_test_selection=true`、`unseen_images_used_for_gradient=false`、`strict_blind_claim=false`。
