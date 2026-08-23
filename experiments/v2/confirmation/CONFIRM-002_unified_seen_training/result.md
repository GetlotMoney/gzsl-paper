# V2-CONFIRM-002 结果

状态：`running_control_queued`。

本实验固定训练50轮，每轮让7,057张seen训练图像各出现一次；不使用三折、阶段式父模型冻结或official-test选模。

RUN-001已完成：

`U=72.066391%`、`S=77.308381%`、`H=74.595407%`、`ZS=79.640341%`，固定报告epoch 50。

每个epoch均记录`sample_count=unique_sample_count=7057`；三个训练模块首批梯度范数均大于0。official test在第50轮checkpoint写入后只加载一次，`test_used_for_selection=false`。

该结果不能直接与旧`H=74.023182%`基线作严格模块增益比较，因为旧训练器每个batch独立重采样，不能保证每轮7,057张各出现一次。下一步只补同协议TG-VPR-only控制，不修改RUN-001参数或结果。

RUN-002控制在看到RUN-001后追加，唯一目的为修复基线训练口径；它不用于重新选择RUN-001的结构、参数、epoch或seed。
