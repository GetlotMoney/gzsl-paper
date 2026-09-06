# V4-TRY-023-R2 Tuned Local-PCLR 单轮双Agent对抗审核 receipt

- 冻结代码：`b0a756dd624e883eb50d19a2455ba06bdc73f118`；tree：
  `8d6260f9f8bb61cd102be8e660731cae5c30b913`。
- Config SHA：`0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49`；
  Parent history SHA：`10591bb35a51949a1989ae3a918b50bca37c1f465a52c6bb5df5552c1b0a4779`；
  relation manifest SHA：`0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4`。
- 唯一R2：Top-15、ridge=`0.03`、cap=`0.5`、correction scale=`2.38`、
  seen-logit gamma=`0.525`。`human_annotations_used=false`、
  `unseen_images_used_for_gradient=false`。
- 共享本地证据：相关`26 passed`，`py_compile`和`git diff --check`通过；两名Reviewer
  未重复测试、资产预载或全量SHA。

Reviewer A/B先独立审完公式、gamma完整200类轴、ZS晚切、Raw/Calibrated Off、beta loss
隔离、成功门、checkpoint/resume和输出合同，再直接交换清单并逐项质询。静态共同结论：
`P0=0/P1=0`，允许共享GPU micro。

GPU0 micro覆盖真实combined parent/relation/beta backward：update1
total/parent/relation/beta loss=`6.960897/3.153720/0.492771/3.314407`，beta=`0.0500004`；
Parent影子参数、loss、CPU/CUDA RNG同轨迹；checkpoint恢复下一步完全一致，update2 state
SHA=`55c3d0ef37e84a6ed7966438be881a11c8d1866f0049eba369e1252750d6653a`。

物理GPU1验证Raw Off的seen/unseen/ZS逐预测canonical parity全部为true；calibrated ZS等于
raw ZS；Full ZS为完整200类校正后晚切；Top-15空诱导子图potential严格0；relation+beta
loss=`3.786259`且梯度有限。gamma=`0.525`，初始effective beta=`0.119`，上限=`0.595`。
环境沿用签字fingerprint SHA
`8b3e2d5d93cdd9763843c3c5f72903f466a86f7524c9dc2b02bb1d4699c32c59`；正式R2输出不存在。

两名原Reviewer直接互认共享micro，最终`P0=0/P1=0`，共同结论：

**代码单轮双Agent对抗审核通过**

剩余P2：effective beta必须持续披露；`pclr_transitions_vs_gtd`表示gamma+PCLR整个R2，纯PCLR
机制只能引用Calibrated-Off；state必须与config合同共同发布；resume NaN与TopK exact tie保留
为低概率健壮性风险。均不阻断唯一R2 Full。

本轮复用了R1资产、Parent history、测试入口和micro脚本，只补R2 gamma/配置差异证据；
未运行controls、重复全量SHA、框架图或额外参数网格。

## 正式RUN结果复核

正式RUN完整完成fixed-150：`21171` updates、`152`个评估点、`150`次teacher refresh，
`stop_reason=completed_fixed_150`。best update=`13818`：

`U/S/H/ZS=79.565275/80.288148/79.925077/87.938917`。

- 相对RUN-030为`+0.855062 H`，距`H=80.070015`和`+1 H`门均差`0.144938`。
- 同checkpoint Raw Off为`+1.023164 H`；同gamma Calibrated Off为`+1.385500 H`。
- `|U-S|=0.722873`，Raw-Off seen/unseen净纠错为`-24/+100`，合计`+76`，ZS安全。
- Raw Off全部`152`点复现RUN-030；metrics SHA为
  `3d64bd36e48304b025044b109c579001279400ccec075fc1246496c4f28e8578`。

两名Reviewer独立复核公式、身份、门槛和结果合同后直接交叉质询，共同结论为
`VALID RUN / GATE FAIL / DROP FINAL / NO RESCUE`，`P0=0/P1=0`。Raw-Off增益和净纠错
包含gamma+PCLR整体；纯PCLR只能引用Calibrated-Off增益及净纠错`+38`。R2已明确是唯一
最终调参补救，后续再选择gamma、scale、Top-K、ridge、cap、checkpoint或门槛均为事后
official-test搜索，不符合预注册合同。
