# V6-TRY-010-R2 / DIAL 代码交叉审查

- 语义代码提交：`6c52d7c63f4457476c286517dc7f0ce410dd3990`，正式父：`52b511d77b4ad048f35b40dc3cbd9afd092167e9`。
- 初始实现：`5c6877eb47bdd1e0c11b2a81a7979dcbc34195d9`；初审冻结账本：`35f4e9eefc8bf8bf7f2fc101073ca5f11552acde`。
- 配置 SHA256：`db6a815e07e1f92904926bbc413ce43199acd81210e708be9fcb2280551af836`。

## 初审与集中修复

两名独立审查均发现同一 P1：旧 directional hinge 只要求 I 不少于所需 margin；所需为零时允许继续同方向过推，违反“只补 S+V 残差”的训练合同。除此之外 P0=0；固定 Top2、反对称 correction、S/V/I off、I 的 13 维 role-patch alignment、梯度隔离、父条件与 Chen-style test 边界均成立。

主 Agent 一次性将 I loss 改为 balanced SmoothL1 拟合 detached 精确有符号 target；目标为零时任何非零 I 输出都会有损失。并添加冻结采样轨迹的正/负/零 target gate 与过推单测。

## 修复复核与一次交叉交流

- A 修复复核：`P0=0/P1=0`，pass；`C:\Users\ADMINI~1\AppData\Local\Temp\DIAL-R2-fix-agentA-review.md@sha256:21518d77ef32ac30bfe9bc4074bf46520acbd7566b31852942ea6cc9969c76d9`。
- B 修复复核：`P0=0/P1=0/P2=0`，pass；`C:\Users\ADMINI~1\AppData\Local\Temp\DIAL-R2-fix-agentB-review.md@sha256:bee2acb97b9272c3a67bdd18fc6f1406a1beee28eb2728a051a43df478b75c81`。
- A 对 B 交叉回复：`P0=0/P1=0`，**双Agent交叉审查通过**；`C:\Users\ADMINI~1\AppData\Local\Temp\DIAL-R2-fix-agentA-cross-reply.md@sha256:8fd33c7e1813cec8e1f34f5d1c2f6884ac5a5fda65a6a741816a8af8c8ef5cb5`。
- B 对 A 交叉回复：`P0=0/P1=0`，**双Agent交叉审查通过**；`C:\Users\ADMINI~1\AppData\Local\Temp\DIAL-R2-fix-agentB-cross-reply.md@sha256:bb79dfa73e919b5afbadc24fdb248dc4a4d2a5916a16d1576b79c5900a117c72`。

最小测试：`14 passed`；该审查只覆盖 R2 当前启用的 forward、loss、采样三态 gate 与同 checkpoint 评估合同。允许启动一次 GPU micro 和一次固定 28,228-update 正式运行。用户仍要求不生成 HTML 图；本次以审查收据记录训练 loss 改动。
