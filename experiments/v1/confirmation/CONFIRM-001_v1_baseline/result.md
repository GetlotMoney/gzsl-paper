# V1-CONFIRM-001 结果

状态：`completed`。

RUN-001 在模型初始化时因类别校验 device mismatch 失败，尚未发生训练 step，失败证据已保留。RUN-002 使用最小修复提交和独立输出目录完成 50 epoch 训练。

RUN-002 最佳结果：`U=72.3584%`、`S=76.2365%`、`H=74.2468%`、`ZS=81.4479%`，best epoch 为 `36`。

该结果使用 `test_selected_inductive_gzsl`，official test H 用于选择最佳 epoch，不能描述为 blind-test。代码相对冻结 V1 只修复模型初始化中的 device identity validation；模型公式、forward、loss、数据、参数和评估语义不变。
