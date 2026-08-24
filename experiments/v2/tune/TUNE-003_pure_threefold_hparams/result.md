# V2-TUNE-003 结果

状态：`planned`。

每折严格执行：100类中的80%图像参与梯度；同100类的20%图像作为val-seen；剩余50类全部图像作为val-unseen。pseudo-unseen图像不进入loss或backward，official test不加载。

RUN-001建立三模块当前超参数基准：`topology_weight=0.1 / max_transport_step=1.5 / max_generator_magnitude=0.2`。选择指标是三个fold同一epoch的mean H，并同时记录min/max/range、mean U和mean ZS。
