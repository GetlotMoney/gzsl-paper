# V3-TRY-013多尺度候选差异两轮代码审核

- 准确TG父commit：`cd30797a5eab3aa6ed28bd04df0b17f413730063`
- 初始代码语义commit：`b7a57971eb6062ee701627d7438672022de15c55`
- Round 1修复代码commit：`c8553db96d9e460f181560855b23b55342f4a3af`
- 服务器preflight平票修复commit：`a76c2497fe393bd2deb790ea3161b27a0330828e`
- 分支：`exp/v3/innovation/innovation-004-multiscale-contrast`
- Round 1：`failed_on_b7a5797; post_fix_recheck_pending`
- Round 1 findings：原实现用幅度比冒充18票同向比例、缺少seen训练证据强度硬拒绝、结果缺少ΔU/ΔS/ΔZS，共3个P1。修复为真实18票计数、`12/18`边界、多数票定方向、seen训练可靠pair强度25分位阈值与四指标增量；补11/18拒绝、12/18通过、少数强反向不能翻转、低强度拒绝、ZS全局ID、精确6尺度和beta0测试。
- Round 2：`initial_parallel_preread_invalidated_by_round1_findings`
- 服务器preflight：修复Round 1后首次服务器专项测试发现随机18票可能`9:9`平票，旧方向默认取正导致候选交换反对称测试偶发失败。正式2/3门槛虽会拒绝9:9，但数学合同不严格；修复为平票方向0、可靠性0并新增显式平票测试，重复20次反对称测试和完整测试通过。该发现发生在重新启动两轮审核前。
- Round 1最终：代码`a76c2497`，无P0/P1，审核通过。
- Round 2最终：代码`a76c2497`、RUN `4f1a0d13`、登记`c1582712`，真实完整资产和TG checkpoint micro-batch通过，结论“无P0/P1，第2轮通过”。
- TRY-013结果：最佳overall为`beta=0`父模型；最佳非零`beta=0.05`得到`U/S/H/ZS=78.105056/74.921691/76.480263/85.842818`、`ΔH=-0.177396`，决策`drop_before_training`。结果SHA=`787bcb7eb0ed73279608e7eb59faaee08ef651b39fbce958b3078347ca89939f`，config snapshot SHA=`30d90cf08103a1c7498dab26bc5c28b20543566a0c48b05b6d73703f1629df0e`。
- 运行许可：只有相关代码树的两轮独立审核完整结束，第二轮明确“无P0/P1，第2轮通过”后，才允许运行TRY-013。配置、队列和结果只走确定性contract校验，不重复代码Agent。
