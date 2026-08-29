# V2-CONFIRM-002 结果

状态：`completed_seen_biased_gain`。

本实验固定训练50轮，每轮让7,057张seen训练图像各出现一次；不使用三折、阶段式父模型冻结或official-test选模。

RUN-001已完成：

`U=72.066391%`、`S=77.308381%`、`H=74.595407%`、`ZS=79.640341%`，固定报告epoch 50。

每个epoch均记录`sample_count=unique_sample_count=7057`；三个训练模块首批梯度范数均大于0。official test在第50轮checkpoint写入后只加载一次，`test_used_for_selection=false`。

该结果不能直接与旧`H=74.023182%`基线作严格模块增益比较，因为旧训练器每个batch独立重采样，不能保证每轮7,057张各出现一次。下一步只补同协议TG-VPR-only控制，不修改RUN-001参数或结果。

RUN-002控制在看到RUN-001后追加，唯一目的为修复基线训练口径；它不用于重新选择RUN-001的结构、参数、epoch或seed。

| Condition | U | S | H | ZS | epoch | test用于选择 |
|---|---:|---:|---:|---:|---:|---|
| RUN-002 TG-VPR-only | 73.326439 | 74.331790 | 73.825692 | 81.534684 | 50 | false |
| RUN-001 unified | 72.066391 | 77.308381 | 74.595407 | 79.640341 | 50 | false |
| unified - control | -1.260048 | +2.976590 | **+0.769715** | -1.894343 | - | - |

结论：共享迁移与CCGR在无三折、无阶段冻结、无test选模条件下提供了真实H增益，但增益来自seen准确率，U和ZS都下降。该结构保留为干净训练信号，不晋级为最终78%框架；下一方法必须直接缓解seen偏置，且不能通过official test挑参数。

模型SHA：

- RUN-001：`2f9833e18c7f8fb07a34c3f2147e675def8fd7e5cd1c84c982df9e14afa03c5c`
- RUN-002：`fdaf300049424c70449784578bf7c35eb689dd74d42a90dab62e58866db07338`
