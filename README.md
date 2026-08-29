# gzsl-paper

面向 CUB、AWA2 和 SUN Attribute Generalized Zero-Shot Learning（GZSL）的干净研究仓库。

## 正式框架

- `FRAMEWORK-V1`：冻结分支`framework/v1`、Tag`v1`，[HTML框架图](experiments/v1/framework_diagram.html)，确认基线H=`74.2468%`。
- `FRAMEWORK-V2`：TG-VPR-H1独立框架，冻结分支`framework/v2`、Tag`v2`，[HTML框架图](experiments/v2/framework_diagram.html)，首个正式单seed基线H=`74.023182%`。
- `FRAMEWORK-V4`：owner晋级的TG+GTD三数据集框架，冻结分支`framework/v4`、Tag`v4`，[HTML框架图](experiments/v4/framework_diagram.html)；CUB/SUN显示GTD正增益，AWA2保留精确no-op边界。
- V1与V2是两套独立训练路径；V2不接入或静默修改V1。
- 三套正式框架代码都只迁入必要实现与来源信息，不继承旧Git历史、旧实验账本或旧研究知识。
- owner已选择`FRAMEWORK-V4 / TG+GTD`作为当前论文主框架；下一创新必须与TG原型学习和GTD unseen几何迁移形成自然衔接。
- V4已索引CUB、AWA2和SUN原始RUN作为晋级证据；跨版本证据保留原路径、原commit和原输出，不复制结果。

## 评估协议

owner选择的论文主结果使用Chen-style test-selected inductive GZSL。三数据集统一协议见[最终实验协议](docs/FINAL_THREE_DATASET_PROTOCOL.md)：

1. 每个数据集使用全部`trainval_loc`图像；unseen图像不进入梯度。
2. 每步独立随机抽50张，总更新数为`ntrain×200//50`。
3. 每`niters//200`步评估official test，并根据整套模型official H保存best checkpoint。
4. U/S/H在该数据集全部seen+unseen类联合空间计算，ZS只在unseen类空间计算。
5. 固定披露`test_used_for_selection: true`，不描述为blind-test。

现有validation-first结果继续作为更严格协议对照。CLIP-based与经典ResNet-101 GZSL分开报告，不能直接混表比较。

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

项目当前任务清单见 [docs/PROJECT_CHECKLIST.md](docs/PROJECT_CHECKLIST.md)。

## 研究闭环

新论文、新证据和新 idea 从本仓库重新建立，不迁移旧 GTPJ 的研究知识。统一规则见 [research/README.md](research/README.md)。

## FRAMEWORK-V2 入口

- 方法说明：[docs/TG_VPR_H1.md](docs/TG_VPR_H1.md)
- 模块代码：`model/frameworks/v2/model.py`
- 独立训练入口：`python -m model.frameworks.v2.train`
- 冻结配置：`config/tg_vpr_h1.yaml`
- 来源身份：`INNOVATION-MODULE-1`
- 历史三数据集候选训练器只保留在对应实验commit的原路径`model/train_paper_v2.py`，不属于当前`main`正式入口。

## FRAMEWORK-V4 入口

- 正式模型代码：`model/frameworks/v4/`
- 晋级来源复现入口：`python -m model.frameworks.v4.train`
- 当前边界：该训练器只接受已冻结的`FRAMEWORK-V3-EXPLORATION`晋级RUN配置；仓库尚未提供新的`FRAMEWORK-V4`基础训练配置，不能把它描述成任意V4新RUN入口。

## checkpoint兼容边界

正式checkpoint保存为`state_dict`或包含`model_state_dict`的字典，目录迁移保持参数键不变。V1额外保留`model.MyModel.GTPJ`旧导入；仓库不承诺加载历史上通过`torch.save(model)`保存的V2/V4完整Python对象pickle。
