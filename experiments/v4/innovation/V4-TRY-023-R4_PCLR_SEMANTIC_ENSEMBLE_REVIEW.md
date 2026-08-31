# V4-TRY-023-R4 PCLR语义ensemble单轮双Agent审核 receipt

- 冻结代码：`c80c021e25bdcef5e5a80e2f01286c9f886e3f52`；tree：
  `8b5d9db326d4dc9be232eeb2d30d8fb8fa9dd2e5`。
- Config SHA：`f9ec1da6074225f947a2ef0d468e1543445bcc7a6df6209a181be025969d98d1`。
- Source R3 config/metrics与R2 model SHA：`8528b715...` / `39bea2db...` / `16b5071f...`。
- 共享本地证据：相关`29 passed`、`py_compile`、`git diff --check`通过。

A/B独立审查后直接交叉发现初始P1：R4未将本次Raw/R3 controls逐指标与source metrics硬
复现。集中修复加入Raw→source Raw Off、R3→source Full两组`U/S/H/ZS`的`1e-6`
fail-closed parity，并分别注入偏差测试；原两名Reviewer复核后P0/P1归零。

物理GPU0/GPU1分别完整复算全部测试图像，metrics逐字节一致，SHA均为
`efbdca19f8248b2e16c99baa7aa5a81d2279218db910a9a00e7303d45d2fc2bc`。R4结果：

`U/S/H/ZS=80.694097/81.446952/81.068777/88.785273`。

六项AND门全部成立：`H>=81`、`ΔParent=1.998762`、`ΔR3=0.882357`、gap=`0.752854`、
ZS安全、Raw seen/unseen net=`129`。Raw/R3 controls exact source；三路finite、ZS晚切、
role0/role6资产、source只读和formal output absent均实证。

两名Reviewer直接互认双GPU micro，最终`P0=0/P1=0`，共同结论：

**代码单轮双Agent对抗审核通过**

R4必须披露`nested_official_test_selection=true`、`strict_blind_claim=false`、
`llm_world_knowledge_used=true`、`human_annotations_used=false`、`expert_attributes_used=false`。
