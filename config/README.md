# 配置入口

现有配置路径已经被Experiment、RUN与仓库外结果绑定，因此本轮目录整理不移动旧配置，也不改写历史`config_ref`。

- `v1.yaml`：FRAMEWORK-V1正式配置。
- `tg_vpr_h1.yaml`：FRAMEWORK-V2正式配置。
- `paper_v2/`：三数据集资产与统一训练配置。
- `tries/`：历史和当前快速尝试配置；文件名中的Vx-TRY身份保持不变。

新配置继续优先绑定准确Framework、Idea和TRY；只有产生新的正式Framework且不会破坏既有引用时，才建立新的版本配置目录。
