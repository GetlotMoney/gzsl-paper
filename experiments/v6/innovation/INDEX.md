# FRAMEWORK-V6-DEVELOPMENT Innovation

`V6-TRY-001 / IDEA-197 / RoleTriPool` 已在 Gate0 失败并保留真实账本。

`V6-TRY-002 / IDEA-198 / SEAV` 已在正式实现前被 No-crop 硬控制证伪：No-crop 点估计高于 Full，因此没有启动正式训练。它从正式V5父提交 `52b511d77b4ad048f35b40dc3cbd9afd092167e9` 独立开始；EAAC与RoleTriPool失败结果只作为前置证据，不是代码父。

`V6-TRY-003 / IDEA-199 / SVRA` 已通过冻结 Gate0：保留 S/V correction-opportunity trigger，删除全部部署期 raw crop，并以4D Parent-risk arbiter完成最终 keep/swap。Full=`68.335831`，相对 Parent/S-off/V-off/I-off 分别为 `+1.642908/+1.764108/+1.484473/+1.642908pp`；所有预注册硬门通过。它仍是待 owner 接纳和 formal/multi-seed 确认的候选，不是正式 `framework/v6`。

随后固定 checkpoint 的 official 诊断显示 sequential SVRA Full `H=62.723317`，低于同骨干 Parent `63.192631`，因此不能晋级。当前快速尝试为 `V6-TRY-004 / IDEA-200 / J-SVRA`：按 owner 要求改为 full200 轴下 S/V/I 真正端到端联合训练，先跑固定1000步 official precheck。

`V6-TRY-004 / J-SVRA` official precheck 已失败：Full `H=56.250432`，Parent `62.441058`，No-joint `60.285100`，Sequential `50.105076`；Full纠正238张但破坏570张。失败原因是加权机会—风险乘积把official trigger率推到41.5%，且leader trigger率高于challenger。

当前快速尝试为 `V6-TRY-005 / IDEA-202 / DESC`：保留端到端 full200 轴和空间辅助监督，但删除概率乘法与正样本加权，改为局部证据直接参与单一 keep-vs-swap logit。

`V6-TRY-006 / IDEA-201 / C-PCLR`由owner从正式V5 commit
`52b511d77b4ad048f35b40dc3cbd9afd092167e9`独立授权。它不继承TRY-001至005代码，固定禁用
Top-17，把冻结关系方向图编译为`G=MD`类别原型；TG/GTD、matched online-V5控制头和C-PCLR
在同一seed7、28,228步内训练，C-PCLR导出`hQ^T+b`。冻结commit `b707b0c...`已完成两名临时
Reviewer的独立清单、直接文件交换和逐项回应，双方`P0=0/P1=0/pass`。当前等待唯一一次GPU
batch50 micro-batch，通过后启动正式fixed-200 RUN。顶层`FRAMEWORK.yaml`仍是未接纳DESC历史
开发元数据，不是本Experiment代码父条件。实验入口见
[`V6-INNOVATION-001_COMPILED_PCLR`](V6-INNOVATION-001_COMPILED_PCLR/EXPERIMENT.yaml)。

V6-TRY-006已完成fixed-200正式RUN：C-PCLR Full `U/S/H/ZS=77.606910/83.639657/
80.510432/88.473403 @ update13818`；S/V/I关闭分别降低`1.350275/1.087737/1.313951 H`，
模块依赖门全部通过。但matched online-V5 best `H=80.818096`，正式V5 `H=81.068777`，
C-PCLR分别低`0.307664/0.558345 H`。正式decision=`drop_gate_b_contract_failed`；不进入Gate C，
不做Top-K、alpha、role、seed或checkpoint补救。

Owner后续明确覆盖内部裁决，将V6-TRY-006保留为效率型论文候选：状态
`owner_override_keep_efficiency_candidate`。原始程序drop、指标差值和未完成的成本证据保持不变；
该覆盖不等于准确率门通过，也不自动晋级FRAMEWORK-V6。

Owner随后将S/V/I统一方法正式晋级为`FRAMEWORK-V7`，论文父基线固定为TG+GTD；内部V5与matched
online-V5继续作为开发历史保留。`FRAMEWORK-V6-DEVELOPMENT`保持另一套待定框架身份，不因V7
晋级被覆盖或合并。
