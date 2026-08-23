# gzsl-paper

面向 CUB Generalized Zero-Shot Learning（GZSL）的干净研究仓库。

## 正式框架

- `FRAMEWORK-V1`：冻结分支`framework/v1`、Tag`v1`，[HTML框架图](experiments/v1/framework_diagram.html)，确认基线H=`74.2468%`。
- `FRAMEWORK-V2`：TG-VPR-H1独立框架，冻结分支`framework/v2`、Tag`v2`，[HTML框架图](experiments/v2/framework_diagram.html)，首个正式单seed基线H=`74.023182%`。
- V1与V2是两套独立训练路径；V2不接入或静默修改V1。
- 两套框架都只迁入必要代码与来源信息，不继承旧Git历史、旧实验账本或旧研究知识。
- owner已选择`FRAMEWORK-V2`作为论文主框架；当前目标为`H >= 77.023182%`并形成三个相互连贯、获得实验支持的创新点。
- 当前核心创新只保留TG-VPR-H1。固定10%保守unseen迁移保留为测试时观察；训练式ELPT正在验证。

## 评估协议

新的论文主结果使用xlsa17类别不相交validation协议：

1. 开发阶段只使用`train_loc`的100类训练，并以类别不相交的`val_loc` 50类选择结构、参数和epoch。
2. validation图像不进入梯度；三折、冻结和多阶段训练允许，但只能发生在开发训练/validation范围内。
3. 选择冻结后，使用`trainval_loc`的150类、7,057张图像重新训练最终模型。
4. official test只在最终checkpoint完成后评估，不用于回改方法。
5. U/S/H在200类联合空间计算，ZS在50个unseen类空间计算。

新主结果记录`validation_used_for_selection: true / test_used_for_selection: false`。历史test-selected结果继续保留，但只能作为探索观察。CLIP-based与经典ResNet-101 GZSL分开报告，不能直接混表比较。

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
- 模块代码：`model/tg_vpr_h1/module.py`
- 独立训练入口：`python -m model.tg_vpr_h1.train`
- 冻结配置：`config/tg_vpr_h1.yaml`
- 来源身份：`INNOVATION-MODULE-1`
