# IDEA-215 / Conditional Residual Grounding (CRG)

- status: `rejected_at_prequeue_gate`
- problem_category: `visual_semantic_interaction`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-214 / NRMP`

## Hypothesis

NRMP与whole-prototype控制持平，说明局部affinity被whole-class/global nuisance支配。CRG先对所有向量单位化，从role中投影掉class mean8 prototype方向，从patch中投影掉正确图像CLS方向，再以role-patch切向残差计算交互。它借鉴DOLG的局部—全局正交融合、Double/Debiased ML的nuisance正交分数、ICML2024联合子空间概念移除与CVPR2020 GZSL冗余消除。

- old_solution_path: raw role-patch affinity dominated by class/global compatibility。
- new_solution_path: remove explicit rank-1 semantic/global nuisance -> evaluate conditional tangent affinity。
- principle_difference: I只使用S/V主方向线性无法解释的残差信息。
- non_equivalence_test: 必须胜过raw、单边残差、no-renorm、wrong-role、shuffled-CLS、global-text与保持范数的随机切向控制。
- minimal_viability: unseen AUC>=0.55、H相对mean8+1、seen/unseen净纠正、控制优势、locality和paired class bootstrap全部通过。
- current_advantage: none；`performance_status=proof_of_path`。
- failure_boundary: rank-1投影可能无法表示真实nuisance，也可能去除有效局部信息。

## 双Agent审查

- 独立A/B：`a2db9a0c864d117c08bba20e3a3dee491f623d4470d5c2dbcf68a31306267db2` / `379d90c580f9bdf175ec505f7a7dd2d17e6099577352e833d63b8364d7264f20`。
- 交叉A/B：`a4da34c50a5ed0fee60bb661db0fc83d00166504c5ac3b6050eed0246cc17b30` / `85c36d05f1a54857800c4ae108acc2143e610986ab7bb23f38e710ccf3a07c15`，proof Gate范围P0/P1=0。
- Gate脚本SHA：`28e4c3100a9e0a12276aaca50afdb38f9d422bcabfb948d917ac118d329b7614`；代码交叉A/B：`87cc83a6eebdfee0f006a79fa5a99587978753fbb0b0fbdcfe60052cfc95a8c6` / `efc650c74825c01a262f314db1302fc8dc87b615d3eaf4de78a584f54ceb1247`。

## Gate结果

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/crg/IDEA-215-GATE0/result.json@sha256:833a95676946d8512db482c76778a487f4293fd7eddd073965d6527ab95ea1af`。

- semantic mean8基线：`H=68.750566`。
- CRG both residual：seen/unseen AUC=`0.510556/0.589407`，但`U/S/H=42.618308/40.881268/41.731720`；纠正/破坏seen=`118/610`，unseen=`213/1007`。
- shuffled-CLS控制：unseen AUC=`0.592561`、`H=41.706220`，与使用正确CLS的CRG相同，否定图像条件global nuisance解释。
- role残差norm中位数`0.2911`、patch残差`0.8909`，低于0.05比例均为0；失败不是近零归一化噪声。
- bootstrap相对semantic H差95%区间约`[-33.36,-21.33]`；相对最强控制包含0。除unseen AUC与norm外全部硬门失败。

## Decision

CRG被否定，不进入模型实现。正交化提高了部分unseen排序，却没有安全决策价值，也不依赖正确CLS方向；它更像高通归一化而非条件交互。下一方向停止把I定义为role-patch定位，改为让S和V各自产生可校准证据分布，再由显式冲突/不确定性状态执行非加法融合。
