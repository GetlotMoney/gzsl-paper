# V3-TRY-022 GTD-TST两轮代码审核

- 准确TG父commit：`cd30797a5eab3aa6ed28bd04df0b17f413730063`
- 继承历史账本commit：`8ea2329191e40d15fb1dcd5aef7fb757f58766a8`
- GTD代码语义commit：`550a36a9483a2c8dec67fc8a8240f259d215b83b`
- 分支：`exp/v3/innovation/innovation-008-gtd-tst`
- RUN配置：`config/tries/v3_try_022_gtd_tst_fixed150.yaml`
- Round 1：`pending`
- Round 2：`pending`
- 审查矩阵：Mean8/Value球面公式、退化与反极边界、CE+theta² oracle、Gate SmoothL1梯度归属、theta0父TG关闭路径、global/local类别轴、seen teacher与true-unseen数据边界、21,171 updates/152评估点、学习率、best/checkpoint/JSON和服务器真实micro-batch。
- 共享证据：本地专项测试、准确diff、config SHA和代码tree hash待冻结RUN commit后生成；服务器资产contract与每条objective的真实GPU micro-batch待两轮审核阶段补齐。
- 运行许可：两轮独立Agent完整结束、所有P0/P1关闭且第二轮明确写出“无P0/P1，第2轮通过”前，禁止服务器smoke、训练和正式RUN。
