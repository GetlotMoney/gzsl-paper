# IDEA-219 / Prototype-Anchored Deletion-Consistent Visual Expert (PADC)

- status: `rejected_at_prequeue_gate`
- problem_category: `visual_representation`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-218 / CCMVE`
- performance_status: `below_parent`

## 假设与路径

PADC冻结8句mean原型，只在36个patch一侧训练rank16残差投影；每个候选类先选最高响应patch，再删除其Chebyshev半径1邻域，从剩余区域选择第二证据。部署视觉分数为两个空间分离证据的均值，并用second-only CE与JS一致性约束，试图在不旋转unseen语义几何的前提下抑制单区域捷径。

- old_solution_path: patch与prototype共同投影，产生互补预测但严重旋转unseen语义几何。
- new_solution_path: 冻结语义锚点 -> 只适配视觉patch -> 删除最强区域 -> 要求空间分离的第二证据仍能分类。
- principle_difference: 学习只移动观察侧，语义侧保持位级不变；局部证据必须通过一次显式删除干预。
- non_equivalence_test: 同时比较Top1、普通Top2同loss、空间Top2仅CE、mean-patch、frozen空间Top2及shared patch+prototype投影。
- minimal_viability: V强度、独有纠正、oracle空间、等权融合、second-only、空间距离、prototype身份与paired-class bootstrap共同过门。
- current_advantage: 无；真实Gate失败。
- failure_boundary: 单边patch投影不能充分对齐冻结文本空间；空间分离不等于语义独立，第二区域仍可能是同一对象或背景。
- why_not_module: PADC未胜Top1和同类控制，不能把删除一致性包装为成立创新。
- paper_level_claim: none。

## 对抗审核与代码审查

- Idea冻结草稿SHA：`b5eddaf2b3f1c66880fd9abdabf9226c551964a216c6776b1d703196f315f560`。
- Idea最终A/B报告SHA：`6a1063cd37789682f9bf9f6685f0c940883ee232f5e9d9b289453cf34cb1d16a` / `031a3a839665b4e134f4315906ff114a9de9c26c6ed612718ec356116790cfd3`，均为P0/P1/P2=0、pass。
- Gate脚本SHA：`b9a3e769c88a3ab7f4fff01c9ae8a10eb493c7d3b949b6df55b145d6fb71fd05`。
- 修复后交叉代码复核：A为P0/P1/P2=0；B为P0/P1=0、P2=3，均明确“双Agent交叉审查通过”。A/B报告SHA：`2451c7da8ef36728294c5ed727558d1534332a2e4910332b6513837729352c8b` / `24b8ab91b03c43f8a780591402a67ec1335b32e390e3479592523909afb26789`。
- 首次启动因缺少`CUBLAS_WORKSPACE_CONFIG`在update0前退出；加入`:4096:8`后原脚本身份不变并完成运行，不属于方法补救。

## 真实结果

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/padc/IDEA-219-GATE0/result.json@sha256:7436e5cd5f63722b73b7636fdd646be6b1bd72be1057d421cf10613457bdfdad`；日志SHA：`746f3c4d9ea007a105fde612be1a32d72ba09458d02966d10673d83a3d456f47`。

| 条件 | U | S | H | ZS |
|---|---:|---:|---:|---:|
| mean8语义S | 69.1363 | 68.3691 | 68.7506 | 86.1468 |
| PADC V | 23.5975 | 42.5376 | 30.3555 | 51.8051 |
| S+PADC等权 | 47.9252 | 61.9533 | 54.0438 | 73.7211 |
| S/PADC oracle | 72.2468 | 75.1464 | 73.6681 | 88.7402 |
| Top1控制 | 25.4686 | 44.3473 | 32.3555 | 54.0122 |
| shared-coordinate控制 | 33.0535 | 51.7436 | 40.3388 | 58.6047 |
| S/shared oracle | 73.7356 | 75.9544 | 74.8286 | 89.6730 |

PADC的second-only为`H=27.8223 / ZS=48.9371`，但主视觉分数低于Top1约2点，也未胜普通Top2和空间CE-only；等权融合比S低14.7068点。它确有互补错误（seen/unseen中S错V对为122/92张），但unseen仅覆盖20类，不能转化为安全增益。

shared-coordinate控制反而达到`H=40.3388 / ZS=58.6047`且oracle相对S为`+6.0780 H`，但prototype cosine mean/min降至`0.5371/0.0114`。这证明下一补救应把S的冻结语义坐标与V的任务坐标分开，而不是禁止V拥有自己的坐标；随后必须学习可靠性融合，不能等权相加。

披露：`test_used_for_selection=true`、`test_used_for_hyperparameter_selection=true`、`nested_official_test_selection=true`、`unseen_images_used_for_gradient=false`、`strict_blind_claim=false`。
