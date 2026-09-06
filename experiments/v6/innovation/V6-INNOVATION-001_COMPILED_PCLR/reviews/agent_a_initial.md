# Reviewer A initial review — V6 C-PCLR

## 结论

- 审查身份：Agent A，阶段1独立代码审查。
- 冻结代码 commit：`b707b0c4671051244cebf4f8404299fc016b281e`
- 准确 base：`52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- RUN config SHA：`73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b`
- 本地已声明测试：`29 passed`
- 结论：`pass`
- P0：0
- P1：0
- P2：4

我没有发现会导致当前 RUN 公式失真、梯度断路、训练预算不匹配、seen/unseen 泄漏、Top-17 违规、S/V/I 关闭口径错误、best-H/best-ZS 拼接或 matched online-V5 控制失效的 P0/P1。

## 本轮读取与核对范围

已读取并核对：

- `AGENTS.md`
- `research/ideas/IDEA-201_compiled_pclr.md`
- `model/frameworks/v6/compiled_pclr.py`
- `model/frameworks/v6/train_compiled_pclr.py`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/configs/RUN-001.yaml`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/EXPERIMENT.yaml`
- `experiments/v6/innovation/V6-INNOVATION-001_COMPILED_PCLR/framework_diagram.html`
- `tests/frameworks/v6/test_compiled_pclr.py`
- `52b511d77b4ad048f35b40dc3cbd9afd092167e9..b707b0c4671051244cebf4f8404299fc016b281e` 的相关 diff

未读取或等待 B 文件；阶段1保持独立。

## 关键不变量审查

### 1. 公式与导出

通过。

- C-PCLR 编译图方向采用 `D = relation[:,0] - relation[:,1]`，二元 incidence 采用 `+1/-1`，并通过 ridge solve 得到 `M=(B^T B + lambda I)^-1 B^T` 等价映射；代码在 `compiled_pclr.py:120` 生成 `compiled_g = mapping @ direction / relation_temperature`。
- 导出 `Q` 时，关系半边使用 `alpha * compiled_g`，关闭 I 时使用同 shape 零张量；见 `compiled_pclr.py:264-273`。
- seen bias 独立作为 `b_seen=-gamma`，unseen bias 为 0；导出 logits 与 forward 语义一致。
- `compiled_g` 是 buffer 且不可训练，参数合同中标记为非训练导出项，见 `compiled_pclr.py:356-364`。

没有发现 `G=MD` 被实现成可学习同分数 head、在线 Top-K 重排或未冻结图表示的证据。

### 2. 梯度到 Reader / alpha / roles / source control

通过。

- C-PCLR head 的分类损失与 relation-direction loss 都从 seen 训练 batch 产生；`relation_direction_loss` 只使用 seen labels 和 seen-seen incident edges，见 `compiled_pclr.py:297`。
- `read_images` 对输入 image feature 执行 detach 后只训练 Reader 残差，避免视觉缓存被反向污染；见 `compiled_pclr.py:251`。
- 测试覆盖 alpha 与 role 权重非零梯度，且覆盖真实 PaperV2ThreeModuleModel 下 inactive grad 的兼容路径。
- 训练 runner 对 source 的 parent/gate 梯度、online-V5 reader/beta 梯度和 C-PCLR head 梯度分别做 receipt；source receipt 对实际 active parameter group 至少要求一个有限梯度，允许组内兼容参数 `grad=None`，见 `train_compiled_pclr.py:458-492`。
- `from_source_model` / `sync_source_prototypes` 已临时切 eval、保存并恢复 RNG 和训练态，避免 prototype 同步扰动训练随机性。

未发现 Reader、alpha、roles 或 source active 训练路径被整体断开的 P0/P1。

### 3. 200-epoch same-budget matched online-V5 control

通过。

- RUN config 强约束 `nominal_epochs=200`、`total_updates=28228`、`batch_size=50`、`eval_interval_steps=141`、`matched_online_v5_control=true`，且 runner 在配置加载阶段硬校验这些值，见 `train_compiled_pclr.py:74-125`。
- training source 不加载 R2 checkpoint 权重作为训练初始化；只用 source config/tensors 以 seed7 构造同源初始化，checkpoint 只用于 parent parity control，见 `train_compiled_pclr.py:166` 附近和 parent-control loader。
- TG/GTD parent、matched online-V5 Reader/beta、C-PCLR head 在同一 `for update in range(1, total_updates+1)` 一段式训练中更新；见 `train_compiled_pclr.py:752`。
- matched online-V5 control 的 best-H 与 C-PCLR best-H 独立记录；online best 更新见 `train_compiled_pclr.py:849-855`，C-PCLR gate 使用 `required_parent_h=max(formal_parent_H, matched_online_v5_best_H)`，见 `train_compiled_pclr.py:879-927`。
- 共享同一个 source parent/gate 训练流，但 online-V5 的 reader/beta 与 C-PCLR head 参数是独立头；同时反传不构成参数复用作弊，因为控制 loss 更新 source reader/beta，C-PCLR loss 更新 C-PCLR head，parent/gate 则作为两者共同的 same-budget backbone。

未发现 C-PCLR 偷用 R2 trained checkpoint 初始化、matched control 少训/多训、或把 formal checkpoint 当作训练起点的 P0/P1。

### 4. 数据边界

通过。

- 梯度路径只使用 `trainval_loc` 的 seen train features/labels，并在 runner 中检查 7,057 张训练图像与 150 个 seen label；official test 只用于 Chen-style selection/evaluation。
- config 明确披露 `test_used_for_selection: true`、`unseen_images_used_for_gradient: false`、`strict_blind_claim: false`。
- U/S/H 使用 200 类联合竞争，ZS 使用 50 unseen 类竞争；runner 分别调用 GZSL 和 ZSL evaluator，未发现跨 checkpoint 拼接。

未发现 unseen test image 进入梯度、用 test unseen 生成训练资产、或把 Chen-style test-selection 伪装成 blind test 的证据。

### 5. Top-17 禁用与 S/V/I 关闭路径

通过。

- C-PCLR 当前 RUN config 中 `candidate_top_k: null`，runner 对该值硬校验；C-PCLR head 内没有在线 Top-K candidate filter。
- matched online-V5 control 仍可按 source config 使用父路径部署逻辑，这是控制条件本身，不是 C-PCLR 的 Top-17。
- S-off 关闭 semantic roles，V-off 关闭 C-PCLR reader 残差，I-off 关闭 compiled relation half 的 alpha 通道；三者都从同一 C-PCLR best checkpoint 评估，见 `_condition_logits` 与 final/gate 记录。
- 新增真实 PaperV2ThreeModuleModel full/off/off 测试覆盖 `semantic_group_logits.grad is None` 而 receipt 仍通过的兼容情形；这支持“关闭路径是真实单模块关闭，不是 receipt 假阳性导致训练不可执行”。

未发现关闭条件在看到结果后改变、使用不同 checkpoint、或把多个模块一起关掉再冒充单模块贡献的代码路径。

### 6. best-H 与 best-ZS

通过。

- C-PCLR best checkpoint 由 Full-H 选择；best-ZS 仅作为独立观察记录，不参与 checkpoint 选择。
- matched online-V5 control best-H 独立选择并保存 state；其 best-ZS 也独立观察记录。
- final 使用 C-PCLR best-H checkpoint 的 Full/S/V/I 指标做 gate，未发现从 best-ZS 或其他 update 拼接 U/S/H/ZS 的路径。
- update0 被禁用：best update 必须大于 0，且 gate contract 测试覆盖该条件。

## P0 清单

无。

## P1 清单

无。

## P2 清单

### P2-1：实验账本仍有 pending refreeze / final-review-pending 占位

- 证据：`EXPERIMENT.yaml` 和 `framework_diagram.html` 中仍有 `pending_refreeze`、`final-review-pending` 类占位。
- 影响：不影响当前代码 RUN 的数学、训练和评估语义；但正式启动/归档前应绑定实际 frozen commit、review 文件和后续交叉审查结论。
- 关闭条件：RUN 前或 post-run 账本提交中记录 `b707b0c4671051244cebf4f8404299fc016b281e`、本 review 文件路径/哈希、B review 和交叉回应路径；不得把 pending 占位当作最终审查声明。

### P2-2：diff 中多个新增文档有 blank-line-at-EOF 警告

- 证据：`git diff --check 52b511d77b4ad048f35b40dc3cbd9afd092167e9..b707b0c4671051244cebf4f8404299fc016b281e` 对若干新增文档报告 blank line at EOF。
- 影响：格式留痕问题，不改变代码路径、配置 SHA 或 RUN 语义。
- 关闭条件：后续纯文档/账本清理时统一规范 EOF；不要为该 P2 单独重启代码审查。

### P2-3：v6 顶层 framework metadata 与当前 C-PCLR experiment 命名不完全一致

- 证据：新增的 v6 顶层框架元数据仍可见早期 DESC / try-005 风格命名，而当前 RUN 位于 `V6-INNOVATION-001_COMPILED_PCLR`。
- 影响：不在 runner 执行路径内，不影响当前 RUN；但后续论文/账本读者可能误读 v6 当前候选身份。
- 关闭条件：若该 metadata 作为正式 v6 框架事实源使用，后续账本提交应明确它是历史/父框架元数据，或更新为 C-PCLR 当前 experiment 的准确引用。

### P2-4：source gradient receipt 是组级证明，不是逐参数白名单证明

- 证据：`_finite_source_gradients` 对每个非空 active parameter group 要求至少一个有限梯度，允许组内兼容参数 `grad=None`。
- 影响：这是合理的兼容实现，已覆盖真实 inactive 参数情形；不构成 P1。但它不能证明组内每个预期训练参数都获得梯度。
- 关闭条件：如果正式审计需要更强证据，可额外记录每个 active group 的参数名、`requires_grad`、`grad is None` 状态和“允许 inactive”的白名单；当前 RUN 不必因此阻断。

## 最强反例

最强可推翻当前候选的反例不是代码层泄漏，而是实验层真实结果：在同 seed7、同 28,228 updates、同 Chen-style official-test selection 下，matched online-V5 control 的 best-H 等于或超过 C-PCLR Full best-H，或任一 S/V/I 单模块关闭的 `H_full - H_off < 1.0`。这种结果会说明 C-PCLR 没有超过同预算在线父路径或缺少独立模块贡献。

当前代码已经把这个反例编码进 gate：`required_parent_H=max(formal_V5_H, matched_online_v5_best_H)`，并要求 S/V/I 三项 delta 都至少 1.0。因此该反例如果发生，会使 RUN 失败，而不是被代码掩盖。

## 关闭条件

当前 P0/P1 关闭条件已满足：

- frozen commit 与实际 HEAD 一致。
- RUN config SHA 与指定 SHA 一致。
- C-PCLR 公式、导出 Q/b、relation loss、Reader/alpha/roles 梯度路径和 parameter contract 自洽。
- 训练预算硬校验为 batch 50、200 nominal epochs、28,228 updates、141-step official eval。
- matched online-V5 control 与 C-PCLR 在同一 seed/source/updates 下训练，且 best-H 独立选择。
- C-PCLR checkpoint 选择只用 Full-H，best-ZS 独立观察。
- seen/unseen 数据边界和 Chen-style disclosure 明确。
- Top-17 对 C-PCLR 禁用。
- S/V/I 在同一 C-PCLR best checkpoint 上分别单模块关闭。

P2 不阻断启动当前可逆研究 RUN；后续账本和正式声明按上方 P2 关闭即可。
