# V2-TUNE-007 结果

状态：`topology_coarse_planned / not run`。

先对CUB、AWA2、SUN依次运行`topology_weight=0/0.03/0.2`；`0.1`复用ABLATION-004相同text-v2资产、Stagewise、seed7和默认参数的正式结果。完成每个数据集粗搜后再按预注册区间追加细搜，不做笛卡尔积。

执行时物理GPU 0与1各运行一条独立单卡RUN；每张卡最多一个进程，RUN之间不共享梯度或输出目录。
