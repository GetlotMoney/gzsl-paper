# gzsl-paper

面向 CUB、AWA2 和 SUN Attribute Generalized Zero-Shot Learning（GZSL）的干净研究仓库。

## 正式框架

- `FRAMEWORK-V1`：冻结分支`framework/v1`、Tag`v1`，[HTML框架图](experiments/v1/framework_diagram.html)，确认基线H=`74.2468%`。
- `FRAMEWORK-V2`：TG-VPR-H1独立框架，冻结分支`framework/v2`、Tag`v2`，[HTML框架图](experiments/v2/framework_diagram.html)，首个正式单seed基线H=`74.023182%`。
- `FRAMEWORK-V4`：owner晋级的TG+GTD三数据集框架，冻结分支`framework/v4`、Tag`v4`，[HTML框架图](experiments/v4/framework_diagram.html)；CUB/SUN显示GTD正增益，AWA2保留精确no-op边界。
- `FRAMEWORK-V5`：owner晋级的TG+GTD+PCLR-RSE框架，冻结分支`framework/v5`、Tag`v5`，[HTML框架图](experiments/v5/framework_diagram.html)；CUB正式`U/S/H/ZS=80.694/81.447/81.069/88.785`。
- `FRAMEWORK-V7`：owner已接纳的TG+GTD+C-PCLR-SVI论文框架晋级候选，正式`framework/v7`与Tag`v7`将在最终审查后一次性冻结，[HTML框架图](experiments/v7/framework_diagram.html)；论文父基线TG+GTD `H=79.070`，完整框架`U/S/H/ZS=77.607/83.640/80.510/88.473`。
- V1与V2是两套独立训练路径；V2不接入或静默修改V1。
- 正式框架代码都只迁入必要实现与来源信息；旧版本实验账本保持原路径只读，不复制或重编号。
- owner已选择`FRAMEWORK-V7 / TG+GTD+C-PCLR-SVI`作为当前论文首个正式框架；论文父框架固定为TG+GTD。`FRAMEWORK-V6-DEVELOPMENT`继续作为另一套待定开发框架保留，不与V7合并。
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

`output-dir` 必须位于 Git 仓库外且事先不存在。每个训练 RUN 至少产生：

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

## FRAMEWORK-V5 入口

- 固定参数与证据：`experiments/v5/FRAMEWORK.yaml`、`config/framework_v5.yaml`。
- 部署logits入口：`model/frameworks/v5/model.py`。
- 正式评估入口：`python -m model.frameworks.v5.evaluate`。
- V5复用已审R2 checkpoint并执行R3 PCLR推理与R4角色语义ensemble；不是重新训练入口。
- 必须披露nested official-test selection、nonblind和LLM可见形态知识使用。

## FRAMEWORK-V7 入口

- 固定参数与证据：`experiments/v7/FRAMEWORK.yaml`、`config/framework_v7.yaml`。
- 独立无图部署入口：`model/frameworks/v7/model.py`。
- 正式评估入口：`python -m model.frameworks.v7.evaluate`。
- 论文方法父框架为TG+GTD；S/V/I分别是角色语义、视觉关系Reader和incidence关系编译。
- 部署只执行Reader与`hQ^T+b`，不依赖V6实验模块、Top-K、关系边或Laplacian求解。
- 必须披露test-selected、nonblind、LLM关系文本使用以及多数据集/成本证据尚未完成。

## checkpoint兼容边界

正式checkpoint保存为`state_dict`或包含`model_state_dict`的字典，目录迁移保持参数键不变。V1额外保留`model.MyModel.GTPJ`旧导入；仓库不承诺加载历史上通过`torch.save(model)`保存的V2/V4完整Python对象pickle。
