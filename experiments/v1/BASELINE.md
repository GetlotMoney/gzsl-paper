# FRAMEWORK-V1 基线

状态：`pending_first_run`。

首个正式基线必须使用 `config/v1.yaml`，训练梯度只使用 7,057 张 seen 训练图像；允许在训练过程中使用 official test U/S/H/ZS 选择最佳 epoch。完成后在本页记录准确 commit、配置 SHA、数据身份、seed、U/S/H/ZS、best epoch 和 Warehouse URI。

当前实验：`V1-CONFIRM-001`，见 `confirmation/CONFIRM-001_v1_baseline/`。RUN-001 在首个训练 step 前因 device mismatch 失败并保留；当前待跑为 RUN-002，服务器输出目录为 `/data/lby/projects/cv_project/GZSL_Warehouse/runs/v1/CONFIRM-001_v1_baseline/RUN-002`。
