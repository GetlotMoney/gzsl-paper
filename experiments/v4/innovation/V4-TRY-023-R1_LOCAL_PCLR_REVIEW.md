# V4-TRY-023-R1 Local-PCLR 单轮双Agent对抗审核 receipt

## 冻结身份与共享证据

- 语义代码commit：`9028fd79c415f3cac670b1644d77403920b1f4e7`；tree：
  `44ab0970701bc0b1955638418a6c75290b6563eb`。
- 初始冻结commit：`d5f59aa2dd60ff903dd0f84bedc887be046d09b5`；交叉发现P1后只做
  一批集中修复，形成上述最终身份。
- Config SHA：`606b0e4d3b69cb3b750d275e13b960bca26025ba72d1a6948964f099b5dd7093`。
- Parent RUN-030 evaluation history SHA：
  `10591bb35a51949a1989ae3a918b50bca37c1f465a52c6bb5df5552c1b0a4779`。
- Relation manifest SHA：
  `0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4`；
  仍为`[438,2,768]`关系向量、`[438,2]`边，`human_annotations_used=false`。
- 共享本地证据：Local-PCLR/GTD相关`24 passed`，`py_compile`与
  `git diff --check`通过；Reviewer未重复整套测试、资产预载或全量SHA。
- 正式输出不存在：审核及micro前
  `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/pclr/V4-TRY-023-R1/metrics.json`
  不存在。

## 独立审查、直接交叉与集中修复

Reviewer A与Reviewer B先对初始冻结身份独立完整审查，再直接交换完整清单并逐项质询。
独立阶段均为`P0=0/P1=0`；交叉阶段共同发现`P1=1`：旧代码只验证module-off最佳点，
却无条件声明152点Parent完整轨迹已复现；旧PCLR的update 18471 Off-ZS偏差正是同型反例。

集中修复只处理该输出合同及直接相关P2：SHA绑定RUN-030完整history；强制152 rows、
evaluation index、update轴和参考四指标有限；initial、逐eval、resume和final均逐点比较Off
`U/S/H/ZS`，全部通过后才允许写`module_off_full_history_reproduced=true`；同时绑定
Local output basename、model-best local config contract、effective beta披露和Off class-id校验。

同两名Reviewer随后并行复核`9028fd79`，直接互认原P1已关闭且没有新P0/P1，允许进入
同一审核轮的共享GPU micro。

## 共享GPU micro

- GPU0执行两步真实combined backward，覆盖Parent CE/topology/GTD gate、relation reader和
  raw beta；update1 total/parent/relation/beta loss分别为
  `6.9720945/3.1537197/0.4927710/3.3256040`，beta=`0.0500004`，全部有限。
- Parent影子模型的参数、loss与CPU/CUDA RNG同轨迹；辅助路径未推进Parent随机序列。
- official GZSL/ZS真实评估有限；Local-Off update1为
  `U/S/H/ZS=66.904950/68.825173/67.851479/86.146760`。
- checkpoint roundtrip后下一步batch、loss与模型完全一致：
  `checkpoint_next_step_equal=true`；update2 state SHA为
  `081a2db5711a1b521fd2ca00e546e958961853a6606c40df16c78f30e82a176a`。
- 物理GPU1完成relation+beta真实前反向，loss=`3.7973449`，梯度全部有限；canonical Off
  的seen/unseen/ZS逐预测parity全部为`true`。
- GPU1构造确定性20节点空诱导子图，both-endpoint mask为空且potential严格全0；关闭态成立。
- 初始effective beta=`0.0625`，effective beta max=`0.3125`，均已显式披露。
- 环境：Python `3.10.20`、PyTorch `2.5.1`、CUDA `11.8`、driver `525.147.05`；
  两张RTX 4090 UUID分别为`GPU-1dca1cb0-d2a2-c075-af6e-a3e9a1eeb968`和
  `GPU-b1df9ad6-832e-1cb8-f096-d49380875928`。

## 最终共同结论与P2

两名原Reviewer对共享micro直接互认，最终`P0=0/P1=0`，共同结论：

**代码单轮双Agent对抗审核通过**

剩余P2不阻断唯一正式Full：

1. 人为损坏且重新提供匹配SHA的resume history若含NaN，actual Off比较没有额外finite检查；
   正常evaluate、atomic checkpoint及expected-resume-SHA不会自然产生，记录为后续健壮性项。
2. Parent Top-K精确并列没有额外class-id tie-break；当前严格确定性环境与固定设备保持同轨迹，
   真实logit精确并列概率低，若发生跨设备复现偏差则RUN直接按身份失败处理。

## 十分钟审计点

本轮超过10分钟，直接原因是A/B交叉发现完整Parent轨迹声明P1，必须停止micro、集中修复并由
原两名Reviewer复核。已删除或后移重复整仓测试、重复资产SHA、controls矩阵、正式框架图和
文档美化；复用同一测试摘要、资产身份和旧checkpoint诊断。P1关闭后只补一份双GPU共享
micro，没有重跑两个Reviewer各自的证据。下次同类模块在首次冻结前即把完整Parent history
SHA和逐点parity作为共享测试入口，避免把best-point parity误写成full-history parity。
