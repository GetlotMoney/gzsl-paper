# V2-CONFIRM-001 参数矩阵

机器事实源是同目录的`PARAMETER_MATRIX.csv`。

| RUN | 阶段 | 条件 | seed | 状态 | U | S | H | ZS | best epoch | 决策 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| RUN-001 | baseline | v2_frozen_default | 7 | completed | 72.655779 | 75.443041 | 74.023182 | 81.534684 | 50 | baseline_recorded |

固定协议：`test_selected_inductive_gzsl`；official test参与结果选择，unseen图像不进入训练梯度。历史H mean=`73.853094%`只作来源参考，不是本RUN基线。
