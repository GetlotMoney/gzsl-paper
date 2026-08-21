# FRAMEWORK-V2 基线

状态：`pending_first_repository_run`。

V2必须使用`config/tg_vpr_h1.yaml`和`python -m model.tg_vpr_h1.train`在本仓库建立正式confirmation基线。完成前不得把迁入的旧多seed摘要描述为V2 confirmed baseline。

来源参考仅为：seeds `5/6/7/8` 的H mean=`73.853094%`、min=`73.709453%`、max=`74.023182%`、range=`0.313729%`。这些数字使用official test做结构选择，只是test-exposed来源证据。

当前正式计划：`V2-CONFIRM-001 / RUN-001`，见`confirmation/CONFIRM-001_v2_baseline/`。运行使用`framework/v2@3dc078c0d52bf358bf24a26e48346c97de9e99ca`、冻结seed `7`和服务器GPU 1；完成前状态仍为`pending_first_repository_run`。

