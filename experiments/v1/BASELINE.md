# FRAMEWORK-V1 基线

状态：`completed_with_runtime_device_fix`。

首个正式基线必须使用 `config/v1.yaml`，训练梯度只使用 7,057 张 seen 训练图像；允许在训练过程中使用 official test U/S/H/ZS 选择最佳 epoch。完成后在本页记录准确 commit、配置 SHA、数据身份、seed、U/S/H/ZS、best epoch 和 Warehouse URI。

首个真实基线来自 `V1-CONFIRM-001 / RUN-002`：`U=72.3584%`、`S=76.2365%`、`H=74.2468%`、`ZS=81.4479%`，best epoch 为 `36`。

- 框架 base commit：`7d842e5c0e5554409eedb3097fea5130a848c9e4`
- 实际 code commit：`f8dd7c72465686cfe4aea8a0f37f658e1176386a`
- config SHA256：`6eb2f663e0a4f26791592c6236febd211fc43341da872647a01bfc02e57f98df`
- data fingerprints SHA256：`e847395cecb651f37ef2114a4afbd8d0b7ca3ee07f9714bafaef9f54e66d7018`
- Warehouse：`/data/lby/projects/cv_project/GZSL_Warehouse/runs/v1/CONFIRM-001_v1_baseline/RUN-002`

实际提交只修复类别覆盖校验的 CPU/CUDA device mismatch；模型公式、forward、loss、数据、参数和评估语义不变。协议仍为 `test_selected_inductive_gzsl`，并明确记录 `test_used_for_selection: true`。
