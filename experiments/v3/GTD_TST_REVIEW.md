# V3-TRY-022 GTD-TST两轮代码审核

- 准确TG父commit：`cd30797a5eab3aa6ed28bd04df0b17f413730063`
- 继承历史账本commit：`8ea2329191e40d15fb1dcd5aef7fb757f58766a8`
- GTD初始代码commit：`550a36a9483a2c8dec67fc8a8240f259d215b83b`
- GTD集中修复代码commit：`4ccf06d25e7d25df02fd96d9d7a1f087f86f573a`
- GTD三态筛选修复commit：`819fec04bb5b6a7b8fb83e637c378bc69e0fb055`
- 分支：`exp/v3/innovation/innovation-008-gtd-tst`
- RUN配置：`config/tries/v3_try_022_gtd_tst_fixed150.yaml`
- 当前审核周期集中修复：筛选结果固定为drop / `trigger_try020_static_below1` / `pending_matched_try020_comparison`三态；后二者均触发并绑定TRY-020，只有第三态`static_support_passed=true`，仍不得在匹配结果前晋级。metrics新增直接`asset_id`。既有150次teacher证据、RNG resume和方法语义不变。
- 本地证据：`tests/test_gtd_tst.py` 7项通过；完整测试`531 passed, 2 warnings, 3 subtests passed`。warnings来自既有SCCC/V1测试，与GTD无关。
- Round 1：`pending_on_819fec0`
- Round 2：`pending`
- 审查矩阵：Mean8/Value球面公式、退化与反极边界、CE+theta² oracle、Gate SmoothL1梯度归属、theta0父TG关闭路径、global/local类别轴、seen teacher与true-unseen数据边界、21,171 updates/152评估点、学习率、best/checkpoint/JSON和服务器真实micro-batch。
- 共享证据：准确diff、config SHA、代码tree hash与本地测试已生成；服务器资产contract、每条objective真实GPU micro-batch、CUDA RNG resume和checkpoint/JSON闭环待两轮审核阶段补齐。
- 运行许可：两轮独立Agent完整结束、所有P0/P1关闭且第二轮明确写出“无P0/P1，第2轮通过”前，禁止服务器smoke、训练和正式RUN。
