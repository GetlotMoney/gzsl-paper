# V2-INNOVATION-062 结果

状态：`rejected_small_pair_overfit`。

首次启动在完成iteration 0评估后的下一次backward失败：预计算pair张量保留冻结父模型计算图，触发`Trying to backward through the graph a second time`。该次记为`failed_runtime`，未产生正式metrics，不计方法预算；原输出目录保留不覆盖。修复为显式detach pair张量并冻结calibrator，使用全新`RUN-001-RERUN-001`目录执行同一配置。

RERUN正常完成。25分位、同族且真类位于top2的train pair仅`169`个，top1真类比例=`0.573964`。pair CE可下降，但所有非零selector条件均降低official H；best严格退回`H=78.320510%`、selected iteration=`-1`、四维权重和bias全0。

IDEA-096拒绝：小pair集对seen过拟合。下一补救改变pair训练语义，纳入所有同族真类top2样本，并按原soft gate加权；因此必须新建Experiment。

模型SHA256：`4da2418753f2650911490d3b514ad458604eda6b1d5e40a35797b01eb9848765`；最后checkpoint SHA256：`237370202ee4c1f714235fd97800b8aa201f93295fda28e4e67d412b926da13f`。
