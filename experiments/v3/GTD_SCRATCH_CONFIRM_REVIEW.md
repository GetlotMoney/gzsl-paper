# V3-TRY-040/041 GTD从头训练确认

- Owner确认base：`b548bfc08acfcf029eaf0b3017b1e5b722aec661`。
- RUN代码提交：`4d46ba1ef8d1c53c0e7fd5c5623f3c56af6dc1b2`。
- TRY-040 config SHA：`b3b04dd71f188acc903b0b0e722eeeb027a15b731a88dee73362f3ffb7d3b469`。
- TRY-041 config SHA：`4a7c4f7385d97a4c0294868cda9f8180eba9c53a55da6f76b27da94b47cd7e2b`。
- 本机资产manifest SHA：`3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6`，从lab4090复制必需的全局CLS、标签和八角色文本张量到仓库外本地Warehouse。
- 初始化：两个条件均不加载TG checkpoint；在seed7下构造同一TG与零初始化GTD scaffold，记录`initial_tg_state_sha256`。
- 唯一条件差：TRY-040的`gate_loss_weight=0`并用TG父原型评估；TRY-041的`gate_loss_weight=1`并用GTD迁移后原型评估。其余训练配置完全相同。
- 训练：batch50、固定150名义epoch/21,171 updates、每141步及尾点official评估，共152点；CLIP图像/文本特征冻结。
- 从头学习率：TG严格复用原V3-TRY-002从头训练的固定`1e-4`；GTD Gate从`1e-5`在前5轮warmup到`1e-4`，之后余弦下降。两条件TG学习率逐update相同。
- 证明目标：两RUN的初始TG SHA、batch序列和TG父梯度必须一致；比较global-best Full指标，并在TRY-041同checkpoint报告GTD-off四指标。
- 本地测试：专项`11 passed`，全仓`535 passed / 2既有warnings / 3 subtests passed`。
- 审核状态：侧对话不能调用独立Agent，本次只允许本机诊断性复现；结果不得冒充已通过项目两轮审核的正式论文证据。
- 无效预跑：`V3-TRY-040-invalid-continuation-lr1e-5`只运行约2个名义epoch后停止；错误沿用了热启动微调LR，不产生正式结果、不计入比较。

## 本地完整结果

两个条件均完整执行21,171 updates、保存152个评估点，global best同时位于update11,280（名义epoch80）：

```text
TRY-040 TG-only： U/S/H/ZS = 77.309537 / 75.999695 / 76.649020 / 86.146760
TRY-041 TG+GTD：  U/S/H/ZS = 79.624420 / 76.670682 / 78.119641 / 85.794950
GTD增量：                     +2.314883 / +0.670987 / +1.470620 / -0.351810
```

- TRY-041同checkpoint GTD-off与TRY-040 global best逐项相同，因此Full-Off同样为`+1.470620 H`。
- 两RUN的`initial_tg_state_sha256`均为`e968092e2a8c2b7b186642bfddf1ebc0864ccb367bc6d5482c5992a6d097f686`。
- 152个对齐评估点中，TRY-041的module-off U/S/H/ZS与TRY-040逐点最大误差全部为0；最终全部TG父参数逐tensor相等，最终teacher package SHA也相同。
- TRY-041 U/S差为`2.953738`，同时通过累计与同checkpoint两项1H门槛，并达到H≥78目标；ZS相对控制下降`0.351810`，需如实披露。
- 该结果使用official test选择global best，`test_used_for_selection=true`、`strict_blind_claim=false`；且尚无两轮独立Agent签字，所以当前结论仅为高可信本地诊断，不是正式论文确认RUN。
