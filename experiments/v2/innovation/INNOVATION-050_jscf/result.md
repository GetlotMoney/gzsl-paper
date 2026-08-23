# V2-INNOVATION-050 结果

状态：`testing_rescue_1`。

RUN-001三个参数组均获得梯度，但所有联合更新条件都低于父模型，best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`、selected iteration=`-1`。mask覆盖均衡，初始权重成功恢复。

训练过程中SEBC gamma从`0.153261`持续升到约`0.1853`且H同步下降，说明普通seen CE破坏了原本由class-exclusive episode学到的竞争偏置。RESCUE-1冻结SEBC，只协调SDRS一维斜率与SDCR八维句权重；其他条件不变。

模型SHA256：`607d73bf03270506f5e803a9ee5247714d739d9ba8f7abfdfdba744240522fdc`；最后checkpoint SHA256：`ae5f0f0897111d4de276a07b2afe8fb82adfc095151c132fa8860357183265e2`。
