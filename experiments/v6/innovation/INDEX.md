# FRAMEWORK-V6-DEVELOPMENT Innovation

`V6-TRY-001 / IDEA-197 / RoleTriPool` 已在 Gate0 失败并保留真实账本。

`V6-TRY-002 / IDEA-198 / SEAV` 已在正式实现前被 No-crop 硬控制证伪：No-crop 点估计高于 Full，因此没有启动正式训练。它从正式V5父提交 `52b511d77b4ad048f35b40dc3cbd9afd092167e9` 独立开始；EAAC与RoleTriPool失败结果只作为前置证据，不是代码父。

`V6-TRY-003 / IDEA-199 / SVRA` 已通过冻结 Gate0：保留 S/V correction-opportunity trigger，删除全部部署期 raw crop，并以4D Parent-risk arbiter完成最终 keep/swap。Full=`68.335831`，相对 Parent/S-off/V-off/I-off 分别为 `+1.642908/+1.764108/+1.484473/+1.642908pp`；所有预注册硬门通过。它仍是待 owner 接纳和 formal/multi-seed 确认的候选，不是正式 `framework/v6`。

随后固定 checkpoint 的 official 诊断显示 sequential SVRA Full `H=62.723317`，低于同骨干 Parent `63.192631`，因此不能晋级。当前快速尝试为 `V6-TRY-004 / IDEA-200 / J-SVRA`：按 owner 要求改为 full200 轴下 S/V/I 真正端到端联合训练，先跑固定1000步 official precheck。

`V6-TRY-004 / J-SVRA` official precheck 已失败：Full `H=56.250432`，Parent `62.441058`，No-joint `60.285100`，Sequential `50.105076`；Full纠正238张但破坏570张。失败原因是加权机会—风险乘积把official trigger率推到41.5%，且leader trigger率高于challenger。

当前快速尝试为 `V6-TRY-005 / IDEA-202 / DESC`：保留端到端 full200 轴和空间辅助监督，但删除概率乘法与正样本加权，改为局部证据直接参与单一 keep-vs-swap logit。

`V6-TRY-005 / DESC` official precheck 已失败：Full `H=43.855002`，Parent `62.441058`，No-action-aux `37.001364`；Full发生1625次swap、净损失840张。空间辅助相对No-action-aux有用，但直接head仍把seen证据尺度误当成可迁移决策信号。
