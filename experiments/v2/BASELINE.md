# FRAMEWORK-V2 基线

状态：`completed_single_seed`。

V2必须使用`config/tg_vpr_h1.yaml`和`python -m model.tg_vpr_h1.train`在本仓库建立正式confirmation基线。完成前不得把迁入的旧多seed摘要描述为V2 confirmed baseline。

来源参考仅为：seeds `5/6/7/8` 的H mean=`73.853094%`、min=`73.709453%`、max=`74.023182%`、range=`0.313729%`。这些数字使用official test做结构选择，只是test-exposed来源证据。

首个当前仓库正式基线来自`V2-CONFIRM-001 / RUN-001`：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`，best epoch为`50`。

- framework/code commit：`3dc078c0d52bf358bf24a26e48346c97de9e99ca`
- config SHA256：`62acb28bb90246bce1ec26c5c5fa02013e5fcbf90fc9bc4b7cc8b45e1421e9b7`
- data fingerprints SHA256：`549886964fbef07bad2f0f65052760e57a34f8b3b26f6efca7795ba3a68d1d8e`
- Warehouse：`/data/lby/projects/cv_project/GZSL_Warehouse/runs/v2/CONFIRM-001_v2_baseline/RUN-001`

该结果逐值复现迁入历史seed 7的H=`74.023182%`，但当前正式基线只有一个seed；它不能单独证明多seed稳定性。
