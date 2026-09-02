# IDEA-212 / Prototype-First Disentangled Evidence (PFDE)

- status: `testing_last_rescue_proof_of_path`
- problem_category: `class_competition`
- mechanism_tags: `eight_sentence_prototype`, `global_patch_visual`, `role_patch_interaction`, `independent_branch_supervision`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-208 / V6-TRY-010`, rescue `3/3`（最后一次）

## 问题

DIAL R2 达到 `H=68.554693`、V gap `+4.439539`，但 I gap 仅 `+0.091948`。原因不是 I 没有梯度，而是旧 V 已经使用 8 个 role query 对36 patch做注意力，提前完成了“语义×视觉”工作；I 只能重复 V。同时旧 S 只把八句话当残差，Top1/Top2 仍由低质量类别名产生。

## 三端与推理

- S：将 6 个部位、1 个全局、1 个独特描述的 8 个归一化文本 embedding 严格等权均值后归一化组成类别原型；CLS 与200类原型直接分类并产生唯一 Top1/Top2。S 另由全局 CLS 的8维 role-difference evidence输出 `d_s`。原型权重冻结，避免seen训练塌缩到单一句。
- V：不使用 role query。36 patch 仅对两个完整类别原型计算 top-3/mean/max 聚合候选证据，输出 `d_v`，负责局部视觉表示增强。
- I：独占 8 个 role-difference query 对36 patch的注意力；将逐角色全局语义证据与逐角色patch证据相乘，只连同基础pair margin与熵输出 `d_i`。I不读取`d_s/d_v`，因此V-off不会连带改变I。

最终只执行一次 `d=d_s+d_v+d_i`，对 Top1/Top2做 `-d/2,+d/2` 反对称修正。没有搜索、crop、hard swap、图或第二轮推理。

S-off 回到 class-name 原型及其 Top1/Top2；V/I 仍按同一函数运行。V-off 只移除整体候选patch聚合，角色注意力 I 保留。I-off 只移除角色×patch对齐，整体视觉 V 保留。额外报告 semantic-only，防止用 S-off 共适应崩溃伪造语义贡献。

## 训练与可证伪合同

Full CE 更新全部可训练分支。三个 balanced pair CE 都从 detached 当前语义原型 base 独立出发：S-only=`base+scatter(d_s)`，V-only=`base+scatter(d_v)`，I-only=`base+scatter(d_i_iso)`；因此每端必须单独学会当前候选对，而不是只等前一端留下残差。I-only 梯度覆盖 role attention query/key 与 interaction MLP，V-only 只覆盖无角色的 visual MLP。

- old_solution_path: 类别名选候选，V先做role-patch交互，I再拟合S+V残差。
- new_solution_path: 八句原型先形成候选；V只做整体patch候选证据；I独占role-patch对应；三端分别受监督后一次合并。
- principle_difference: 先把三端的学习对象拆开，再检验互补，而不是让 V/I 对同一角色证据竞争。
- non_equivalence_test: V-off必须保留I的role attention，I-off必须保留V的整体patch证据；micro中三项辅助loss只能进入各自参数组。
- minimal_viability: 八句均值原型真实 `H=68.750566`、seen/unseen Top2覆盖 `0.808957/0.812605`；Top2 oracle `H=81.328356`，尚有 `+12.577790 H`空间。CUDA micro需证明三路梯度隔离、V/I pair不变、S-off切换到name pair、attention非均匀且test未加载。
- minimal_falsification: 固定 seed7、batch50、28,228 updates，一次 Chen-style official-test-selected运行。Full必须高于同checkpoint semantic-only，且 S/V/I gap全部 `>=1.0 H`；不要求H80。
- current_advantage: 只读gate证明 prototype candidate coverage与oracle空间；`performance_status=proof_of_path`。
- failure_boundary: 整体patch聚合可能丢小部位；严格等权文本不能自适应描述质量；即使功能拆开，S/V/I仍可能纠正同一批pair错误。

没有教师、蒸馏、专家属性、人工属性答案、未见图像梯度或PCLR在线推理。若R3失败，IDEA-208/CTPM按三次救援预算永久关闭，不再调参。

## R3 正式结果与路线关闭（2026-09-02）

固定28,228 update完成，best update `1,692`：`U/S/H/ZS=69.309634/68.619043/68.962609/86.114001`。同checkpoint S/V/I gap=`+6.444643/+2.387763/+0.212043`；Full仅比八句semantic-only `H=68.750566`高`+0.212043`，I未达到`+1`，`module_success=false`。

结果：`metrics.json@sha256:670ee1145495c9cca52683e247573e418d480dea58b722573432804b9ae6656d`；模型：`model_best.pth@sha256:e3a746febe8ba2caeab1f51a2bde456ffcd10ada20c712af2f8ab9125b4c5204`；历史：`evaluation_history.json@sha256:688364cb55adec41689a6b6268c7e3b32aa98eb957cecc0f3ac0d3e18e871bca`。

CTPM三次救援全部耗尽并关闭。最终经验：八句原型本身提供`+5.558 H`，整体patch V可形成约`+2.39`关停差，但三个分支都输出同一加法margin、并用同一pair标签学习，导致I无法形成不可替代的结构化决策对象。下一Idea必须把I表示改为保留角色—区域指派结构的候选验证器，而非第四个margin/head/loss变体。
