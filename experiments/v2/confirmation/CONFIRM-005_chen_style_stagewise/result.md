# V2-CONFIRM-005 结果

状态：`rescue2_rejected_rescue_budget_exhausted`。

固定50/100/50名义epoch三阶段；阶段边界不根据test移动，阶段间使用最后权重继续训练。全程只有一个跨阶段的整模型best-H，不为每阶段分别保存test最大父checkpoint。

正式结果：`U=71.105868%`、`S=80.453974%`、`H=75.491628%`、`ZS=82.531631%`。整模型best位于iteration `7614`、名义epoch `54`、阶段`TRANSFER_CCGR`。

相对V2-CONFIRM-004无专家端到端`H=74.933940%`提高`+0.557688`个百分点，证明分阶段优化有效；但仍低于`H >= 77.023182%`目标，因此不晋级为最终无专家方案。

三阶段均完成预注册边界和总计28,228次更新，共201个official评估点；`nested_official_test_selection=false`。阶段2冻结TG-VPR并将其排除在optimizer之外，计算边界有效。

审计限制：RUN代码在阶段切换时没有清空冻结参数上一阶段遗留的`.grad`字段，因此`metrics.json`中`TRANSFER_CCGR.tg_vpr=1.443447`是旧梯度缓存，不代表TG被更新。活动模块梯度、模型结果、checkpoint和optimizer边界不受影响；后续代码已增加阶段切换全模型`zero_grad(set_to_none=True)`并新增回归测试。

模型SHA：`b495826d58c5fc421c4757d2c9105148b3fc23d310e04ab70fa4dbfdf03c8c1d`。

失败诊断：RUN-001 best时平均迁移步长为`0.444078`，阶段2末增至`0.829518`并伴随U从`71.105868`降至`65.729409`。RESCUE-1仅把`max_transport_step`从`1.5`降到`0.5`，其余条件不变。

RESCUE-1结果：`U/S/H/ZS=74.326867/77.764529/76.006848/82.930040%`，best位于TRANSFER_CCGR阶段iteration 8037/epoch 57。相对RUN-001提高H `0.515219`，相对端到端提高`1.072908`，但距离77.023目标仍差`1.016334`。

阶段2在0.5上限下后半输出几乎固定，说明该上限抑制过迁移但形成硬饱和；下一补救测试中间上限0.75。

RUN-002模型SHA：`4231aba956c3c0ff57a1ac859a6a8748131e2275efcf3bfb63fcced54b32aa99`。

RESCUE-2固定只把迁移步长上限从0.5调至中间值0.75；其余代码、阶段边界、seed、loss和Chen-style选模完全一致。

RESCUE-2结果：`U/S/H/ZS=73.191816/78.666651/75.830543/83.131742%`，低于0.5方案`76.006848`，因此淘汰并关闭步长参数轴。模型SHA：`889dc4533ae45d915b192bc1ef96c2d666be416941eba77dd931e0f48d50ad91`。

V2-CONFIRM-005已完成父条件加两次有效方法补救；下一方向改变阶段2loss语义，必须新建Experiment而不是继续本目录。
