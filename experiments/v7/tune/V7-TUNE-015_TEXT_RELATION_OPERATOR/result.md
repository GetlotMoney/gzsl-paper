# V7-TUNE-015 Text Relation Operator

状态：`completed / drop`（低于 formal V7 门）。

唯一改动：删除逐图 Reader；学习共享 identity-residual 低秩算子 `D_vis = normalize(D_text + A(D_text))`（`A` 零初始化时逐边 `D_vis` 精确等于原一文本 `D_text`，避免 `normalize(0)`），只用 trainval seen 视觉中心差分监督 seen-seen 边方向；同一 `A` 外推到全部 seen/unseen 相关边并经 ridge 编译为单个类别矩阵 `Q`，推理导出仅 `q,bias`。正式 RUN 使用冻结 commit `78de0c812a930eb744974f1d0a354af2cb161174`、config SHA `3e3f3436f8dc10967f5c5227552241d64f1770c1e6e558cb19494ad37c6513f5`，从 seed7 原始初始化训练 28,228 updates 并完成 201 次 official 评估。

## 最佳结果

- best update：`18753`
- U/S/H/ZS：`78.273934 / 80.636215 / 79.437516 / 87.433225`
- 相对 formal V7 `80.510432`：`-1.072915 H`
- 相对 TUNE013 seen-only CE `79.945797`：`-0.508281 H`

同 checkpoint 推理诊断（Full−off，仅作机制诊断，不替代从头重训消融）：i_off `+0.107266 H`（H=79.330250）、s_off `+0.864793 H`（H=78.572723）。

独立 best-ZS 观察（非 best-H 同一 checkpoint）：ZS=87.647575 @23970（该点 H=77.576627, S=85.150546, U=71.240014）。

## 判断

文本关系视觉化算子失败：Full H 低于 formal V7 `1.072915`、低于 TUNE013 `0.508281`。删除逐图 Reader 后训练可收敛但准确率显著下降，说明共享低秩算子无法替代逐图实例读出，且 seen 视觉中心差分监督不足以迁移到 unseen 竞争。程序 decision=`drop_tune015_contract_failed`，停止此方向，不在该失败代码上继续堆叠。

## 产物

- output：`/data/lby/projects/cv_project/GZSL_Warehouse/tune/v7/V7-TUNE-015_TEXT_RELATION_OPERATOR/V7-TUNE-015-CUB-TEXT-RELATION-OPERATOR`
- metrics SHA：`65f8d67a4356c63f79dd7cf8d111468d0d9bfb7353d6dd9665f55d688c679d39`
- model SHA：`c3330fdd81d7b963cf4b1ae394ac6565bdecd152d1a0008a2f208947f532d5f2`
- history SHA：`c99bd3c4bc4338741eed7e86eb2601049164f631df002d87eda73e50022f304b`
