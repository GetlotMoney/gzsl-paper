# V7-TUNE-014 结果

状态：completed / drop（低于 formal V7 门）。

唯一改动：相对 TUNE013，Reader 与 alpha 不再从普通 seen 分类 CE 取得正式梯度，改由三折 class-disjoint pseudo-unseen outer CE 更新（一阶近似：inner 只在临时 head 副本上做一步，不构建二阶 meta 图，`meta_second_order=false`）；TG/GTD/S 仍普通 seen 训练，推理仍一文本 S/V/I 的 200 类冻结前向；不使用真实 unseen 图像梯度。正式 RUN 使用冻结 commit `1c99994ee8454129af398b1c575ef5a6b2a9e617`、config SHA `048b9708861463d15c288396654b9a51b870feed047bdb1919dd6f4767296187`，从 seed7 原始初始化训练 28,228 updates 并完成 201 次 official 评估。

## 最佳结果

- best update：`10152`
- U/S/H/ZS：`79.017454 / 81.291783 / 80.138486 / 87.642199`
- 相对 formal V7 `80.510432`：`-0.371946 H`
- 相对 TUNE013 seen-only CE `79.945797`：`+0.192688 H`

同 checkpoint 推理诊断（Full−off，仅作机制诊断，不替代从头重训消融）：s_off `+2.647120 H`（H=77.491365）、v_off `+1.343053 H`（H=78.795433）、i_off `+1.058027 H`（H=79.080459）、role_shuffle `+1.296172 H`、signflip `+4.626016 H`。

独立 best-ZS 观察（非 best-H 同一 checkpoint）：ZS=88.086027 @24675（该点 H=76.644720, S=85.306007, U=69.580114）。

## 判断

类别留出 outer CE 提供正的小信号：Full H 相对 TUNE013 提升 `+0.192688`，且 U/S 更平衡（U=79.017454 高于 formal V7 的 77.606910），说明把 V/I 训练信号从“普通 seen 分类 CE”换成“类别留出迁移”是有帮助的训练机制候选。

但 H 仍低于 formal V7 `0.371946`，未过 80.510432 门；提升不足 1H，不能仅凭 Full 提升直接宣称 V/I 有效。若继续，需对 TUNE014 做正式 V/I/VI 从头重训消融；当前只作为小的训练机制候选保留，不登记为创新。程序 decision=`drop_tune014_contract_failed`。

## 产物

- output：`/data/lby/projects/cv_project/GZSL_Warehouse/tune/v7/V7-TUNE-014_CLASS_HELD_OUT_VI/V7-TUNE-014-CUB-CLASS-HELD-OUT-VI`
- metrics SHA：`b5b36377df283b4f1afe8542ef53f33394fd6d0d5e27ea5b5562bb6e65408bad`
- model SHA：`961352fc37466eb7e3a37f224a141a68c8df0840cf9d118e31584f7ca3ff9fde`
- history SHA：`515c657c80c877e7b48f95aa5f4664fdf748acd9ab307772de4a2a2d6717b612`
