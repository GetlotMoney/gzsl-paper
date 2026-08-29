# V3-TRY-022 GTD-TST两轮代码审核

- 准确TG父commit：`cd30797a5eab3aa6ed28bd04df0b17f413730063`
- 继承历史账本commit：`8ea2329191e40d15fb1dcd5aef7fb757f58766a8`
- GTD初始代码commit：`550a36a9483a2c8dec67fc8a8240f259d215b83b`
- GTD集中修复代码commit：`4ccf06d25e7d25df02fd96d9d7a1f087f86f573a`
- GTD三态筛选修复commit：`819fec04bb5b6a7b8fb83e637c378bc69e0fb055`
- GTD drop匹配边界修复commit：`fee829bd56b6ac9b366da82e1438b9d7bee872a8`
- 分支：`exp/v3/innovation/innovation-008-gtd-tst`
- RUN配置：`config/tries/v3_try_022_gtd_tst_fixed150.yaml`
- 当前审核周期集中修复：drop结果固定`matched_comparison_required=null / trigger=false / static=false`；仅0.8H以上且gap合格的后二态触发并绑定TRY-020。IDEA失败条件同步为只有低于0.8H或gap失败才直接drop。TRY-020仍是条件触发项，执行前必须单独冻结准确config/RUN/queue，禁止事后定义。
- 本地证据：`tests/frameworks/v4/test_gtd_tst.py` 7项通过；完整测试`531 passed, 2 warnings, 3 subtests passed`。warnings来自既有SCCC/V1测试，与GTD无关。
- Round 1：`pending_on_fee829b`
- Round 2：`pending`
- 审查矩阵：Mean8/Value球面公式、退化与反极边界、CE+theta² oracle、Gate SmoothL1梯度归属、theta0父TG关闭路径、global/local类别轴、seen teacher与true-unseen数据边界、21,171 updates/152评估点、学习率、best/checkpoint/JSON和服务器真实micro-batch。
- 共享证据：准确diff、config SHA、代码tree hash与本地测试已生成；服务器资产contract、每条objective真实GPU micro-batch、CUDA RNG resume和checkpoint/JSON闭环待两轮审核阶段补齐。
- 运行许可：两轮独立Agent完整结束、所有P0/P1关闭且第二轮明确写出“无P0/P1，第2轮通过”前，禁止服务器smoke、训练和正式RUN。
- 正式结果：best update846，`U/S/H/ZS=80.559021/75.587094/77.993901/86.611593`，静态`ΔH=+1.336242`，decision=`pending_matched_try020_comparison`。
- 结果SHA：config=`71e556ff...`，metrics=`4a47de85...`，history=`0bdf2f79...`，teacher-history=`b542489c...`，model=`c45cddfe...`。
