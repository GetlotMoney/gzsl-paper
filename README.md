# gzsl-paper

面向 CUB Generalized Zero-Shot Learning（GZSL）的干净研究仓库。

## 当前框架

- 正式框架：`FRAMEWORK-V1`
- 冻结 Git 分支：`framework/v1`
- 冻结 Tag：`v1`
- 代码来源：旧 GTPJ 的 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b`
- 迁移方式：只复制必要代码，不迁移旧 Git 历史、旧实验或旧账本
- HTML 框架图：[experiments/v1/framework_diagram.html](experiments/v1/framework_diagram.html)

## 评估协议

项目使用 `test_selected_inductive_gzsl`：

1. 训练梯度只使用 CUB `trainval_loc` 的 150 个 seen 类、7,057 张图像。
2. test-seen/test-unseen 可在训练过程中反复评估，并用于选择 epoch、参数和模型。
3. test 图像不进入反向传播；unseen 类图像从不参与梯度训练。
4. U/S/H 在 200 类联合空间计算，ZS 在 50 个 unseen 类空间计算。

该协议在图像监督上属于 inductive GZSL，但不是 blind-test 评估；论文或对外表述必须明确写 `test_used_for_selection: true`。

## 运行

```powershell
conda run -n dvsr_gpu python train.py `
  --config config/v1.yaml `
  --output-dir D:/path/to/GZSL_Warehouse/runs/v1/RUN-001
```

`output-dir` 必须位于 Git 仓库外且事先不存在。每个 RUN 至少产生：

- `training.log`
- `metrics.json`
- `model_best.pth`
- `checkpoint_last.pth`
- `data_fingerprints.json`

实验结构和多 RUN 规则见 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)。

## 研究闭环

新论文、新证据和新 idea 从本仓库重新建立，不迁移旧 GTPJ 的研究知识。统一规则见 [research/README.md](research/README.md)。

## 独立创新模块

- `TG-VPR-H1`已按owner明确授权迁入[正式方法、代码与配置](docs/TG_VPR_H1.md)。
- [HTML框架图](docs/TG_VPR_H1_framework_diagram.html)绑定模块代码提交`4a063218989c6e193a7aa3c593bb4f0f8ecb7379`。
- 当前模块仍是standalone代码身份，尚未接入`FRAMEWORK-V1`，也未占用`V1-INNOVATION-001`。
