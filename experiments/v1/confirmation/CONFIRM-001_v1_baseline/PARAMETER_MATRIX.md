# V1-CONFIRM-001 参数矩阵

机器事实源是同目录的 `PARAMETER_MATRIX.csv`。

| RUN | 阶段 | 条件 | seed | 状态 | U | S | H | ZS | best epoch | 决策 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| RUN-001 | baseline | v1_default | 5 | failed_pre_training |  |  |  |  |  | rerun as RUN-002 |
| RUN-002 | baseline | v1_default_device_validation_fix | 5 | completed | 72.3584 | 76.2365 | 74.2468 | 81.4479 | 36 | baseline_recorded |

固定协议：`test_selected_inductive_gzsl`；允许 official test H 选择最佳 epoch，unseen 图像不进入训练梯度。
