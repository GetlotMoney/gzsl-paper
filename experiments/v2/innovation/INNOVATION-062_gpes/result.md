# V2-INNOVATION-062 结果

状态：`planned`。尚无正式结果。

首次启动在完成iteration 0评估后的下一次backward失败：预计算pair张量保留冻结父模型计算图，触发`Trying to backward through the graph a second time`。该次记为`failed_runtime`，未产生正式metrics，不计方法预算；原输出目录保留不覆盖。修复为显式detach pair张量并冻结calibrator，使用全新`RUN-001-RERUN-001`目录执行同一配置。
