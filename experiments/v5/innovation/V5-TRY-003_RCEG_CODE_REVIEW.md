# V5-TRY-003 RCEG 冻结代码双Agent交叉审查

review_date: 2026-09-01
idea_id: IDEA-189
reviewed_code_commit: `54a3b489b5ba85aa7ad16d3912021cb29b58045a`
fix_commits: [`04e67d2fd699432deabc94902276fe636cae56aa`, `15541190238f4bf12c8fd4509911024ec503dcce`]
final_reviewed_code_commit: `15541190238f4bf12c8fd4509911024ec503dcce`
review_agents: [`/root/rceg_code_a`, `/root/rceg_code_b`]
review_scope: [`model/frameworks/v5/rceg.py`, `model/frameworks/v5/rceg_data.py`, `model/frameworks/v5/train_rceg.py`, `model/frameworks/v5/evaluate_rceg_dev.py`, `tools/prepare_rceg_assets.py`, `tests/frameworks/v5/test_rceg.py`]

## 共享不变量与最小证据

- Git父条件固定为`FRAMEWORK-V5@52b511d77b4ad048f35b40dc3cbd9afd092167e9`；CEC失败代码不是父基线。
- Gate训练只读取100个dev-seen类/4,702张图和100类文本；冻结评估才读取50个dev-unseen类/2,355张图与完整150类文本轴。
- official test不加载；无unseen图像梯度；无dev-unseen文本梯度；无PCLR在线推理。
- target固定为原图冻结`visual.conv1` 1024维patch均值；masked predictor无法读取target位置token；Target-free训练loader物理不打开target文件。
- Full、S-off、V-off、I-off同checkpoint；Absolute-role、Reference-difficulty、Target-free、Target-shuffle、Role-shuffle为预注册独立控制。
- 本地专项测试：初始冻结commit为`7 passed`；修复冻结commit为`8 passed`。

## 双方独立初审

Agent A初始报告：`P0=0 / P1=0 / P2=3 / pass_with_P2`。P2包括S-off仍计算后丢弃role evidence、本地未跟踪文件会触发clean gate、未使用import。

Agent B初始报告：`P0=0 / P1=1 / P2=2 / revise`。P1为S-off虽然数值返回parent，但仍物理读取并计算role-conditioned evidence，违反模块关闭合同；monkeypatch `role_evidence`即可证伪。P2为缺少物理关闭回归测试和服务器磁盘体量预检。

## 一次直接交叉

- Agent A接受B的严格口径，把S-off问题升级为P1；最终改判`P0=0 / P1=1 / P2=2 / revise`。
- Agent B维持P1，并同意本地clean gate与未使用import均非阻断。
- 共同结论：初始冻结commit不得启动服务器RUN；主Agent只集中修复S-off物理关闭及对应测试。

## 集中修复与复核

- `04e67d2`先新增S-off不得调用role evidence的测试与视觉分支；`1554119`补齐语义入口`name_chunk`，使S-off不构造、不读取`role_query/role_triplet/role_embeddings`。
- 回归测试将`role_evidence` monkeypatch为一调用即失败；S-off仍严格`score=0`且`logits=parent`。
- Agent A修复复核：`P0=0 / P1=0 / pass`。
- Agent B修复复核：`P0=0 / P1=0 / pass`，并明确写出“**双Agent交叉审查通过**”。
- 剩余P2仅为重复不可达`elif mode == "s_off"`和运行前磁盘空间确认；均不改变语义，不阻断可逆Gate运行。

## 最终结论

**双Agent交叉审查通过。**

该签字只覆盖上述冻结代码身份。资产manifest SHA、训练/评估config SHA、服务器环境/GPU fingerprint和最终RUN commit需在真实运行前后继续绑定；任何module、forward、loss、数据、资产生成或评估语义变化都会使本签字失效。
