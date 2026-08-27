# V3-TRY-012频域前景代码两轮对抗记录

- 初始实现commit：`da06d83fcd2662ebcbbade5f39d3feeb8401cbc5`
- Round 1修复代码commit：`322874567da88292bab9f94a66edc14576f53700`
- Round 1复核修复代码commit：`e3be1bb42b9caca91a98c58437b2b1793e19b8dd`
- Round 2发现ZS索引修复代码commit：`1bf0c3b2ceb8d5da09b6ee9d56125cb46438bd9b`
- 首次真实资产写盘修复代码commit：`ea802ec7d25a2a367d0f7ea5b58d051fc9ad92c3`
- 准确TG父commit：`cd30797a5eab3aa6ed28bd04df0b17f413730063`
- 分支：`exp/v3/innovation/innovation-003-frequency-foreground`
- 分支来源：Git reflog证明分支在TG父commit创建，随后仅cherry-pick公共规范和历史账本；实现前`model/`、`tools/`相对TG父commit无差异，未继承CLPR及其他失败候选代码。
- Round 1 status：`passed_on_21b27d2; P0=0; P1=0; P2=0`
- Round 1 findings：首次审查发现配置/RUN边界、论文与仓库公式、`ln_post`位置、资产身份门禁和分支来源措辞共5个P1；首次复核发现TG运行资产与官方频域父CLS资产被错误合并为同一路径的1个P1。修复后增加双资产真实加载、运行时CLIP身份、Gaussian核、无合格非零条件安全drop测试。
- Round 2 status：`passed_on_21b27d2; no_P0_P1; 第2轮通过`
- Round 2 findings：首次Round 2发现ZS在50类子集内重排后未恢复到0–199全局类别ID，导致beta=0父ZS错误为0；修复后显式映射回全局ID并新增ZS逐样本transition关闭验证。新Agent对`6580602`给出“无P0/P1，第2轮通过”。随后首次真实资产生成在首batch写盘时发现频域token仍携带`visual.proj`梯度，`.numpy()`被PyTorch拒绝；未生成正式资产目录，失败日志保留于仓库外。修复为资产聚合全程`no_grad`并在写盘前显式`detach`，新增可训练projection真实NumPy写盘测试；该代码改动使旧签字再次失效。
- 资产生成：准确commit `21b27d2`完成正式原子生成；manifest SHA=`8bbb71e11354af1c65134dac4ff623451c4cd56ba6179df9844066415fa2a462`，asset_id=`cub-clip-frequency-18f087134a402a4e`，counts=`7057/1764/2967`。一次手输截短父SHA被生成器在读取图像前拒绝，未生成目录；随后使用完整SHA成功，两个失败日志均保留且不覆盖。
- 诊断配置：`config/tries/v3_try_012_frequency_foreground_diagnostic.yaml`绑定真实TG base、频域manifest、官方父CLS和TG checkpoint；实际运行前需完成针对真实manifest与配置的两轮复核。
- 诊断Round 2账本修复：首次诊断Round 2发现队列仍把TRY-012绑定到资产代码`21b27d2`，但该commit不包含诊断配置，判定P1。队列改为绑定已冻结且包含配置的实际运行commit `f3aefd92c7aa59fd43c617ee13b00486522368ef`；后续提交只更新账本，不改变`f3aefd9`中的代码、配置或资产身份。实际运行必须在clean detached `f3aefd9`执行并传入同一`--expected-commit`。
- 诊断Round 2最终结论：登记commit `a453c33`与实际运行commit `f3aefd9`分离边界通过独立复核，结论“无P0/P1，第2轮通过”，批准clean detached `f3aefd9`运行。
- TRY-012结果：最佳overall严格为`beta=0`父模型；最佳非零为`repo/K=1/beta=0.05`，`U/S/H/ZS=78.637934/74.649912/76.592046/86.273980`、`ΔH=-0.065613`，决策`drop_before_training`。结果文件SHA=`3200f4384ec7067f8c2589be39a207514fb69fca13b0e0a56329d67ccd084c3a`，config snapshot SHA=`bae77b2442fa3c35a1ecadd1f278fc6963c097cc9121f281652f96b6e0179c65`。
- 运行许可：资产生成预注册不占RUN队列；第一轮P0/P1全部关闭、第二轮独立Agent对准确post-fix资产生成commit明确“无P0/P1，第2轮通过”后，才允许生成频域资产。资产生成后另建绑定真实manifest的可执行V3-TRY-012配置并再次审核，才允许诊断。
