# V2-CONFIRM-005 结果

状态：`rescue1_transport_cap_planned`。

固定50/100/50名义epoch三阶段；阶段边界不根据test移动，阶段间使用最后权重继续训练。全程只有一个跨阶段的整模型best-H，不为每阶段分别保存test最大父checkpoint。

正式结果：`U=71.105868%`、`S=80.453974%`、`H=75.491628%`、`ZS=82.531631%`。整模型best位于iteration `7614`、名义epoch `54`、阶段`TRANSFER_CCGR`。

相对V2-CONFIRM-004无专家端到端`H=74.933940%`提高`+0.557688`个百分点，证明分阶段优化有效；但仍低于`H >= 77.023182%`目标，因此不晋级为最终无专家方案。

三阶段均完成预注册边界和总计28,228次更新，共201个official评估点；`nested_official_test_selection=false`。阶段2冻结TG-VPR并将其排除在optimizer之外，计算边界有效。

审计限制：RUN代码在阶段切换时没有清空冻结参数上一阶段遗留的`.grad`字段，因此`metrics.json`中`TRANSFER_CCGR.tg_vpr=1.443447`是旧梯度缓存，不代表TG被更新。活动模块梯度、模型结果、checkpoint和optimizer边界不受影响；后续代码已增加阶段切换全模型`zero_grad(set_to_none=True)`并新增回归测试。

模型SHA：`b495826d58c5fc421c4757d2c9105148b3fc23d310e04ab70fa4dbfdf03c8c1d`。

失败诊断：RUN-001 best时平均迁移步长为`0.444078`，阶段2末增至`0.829518`并伴随U从`71.105868`降至`65.729409`。RESCUE-1仅把`max_transport_step`从`1.5`降到`0.5`，其余条件不变。
