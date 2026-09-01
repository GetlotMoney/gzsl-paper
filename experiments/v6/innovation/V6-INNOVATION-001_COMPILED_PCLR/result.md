# V6-INNOVATION-001 / V6-TRY-006 result

状态：`review_independent_pass_cross_blocked`。两名Reviewer对commit `89b2908a388c4c6586bbf19216fd50fc777ffdb3`
均独立给出`P0=0/P1=0/pass`，但临时子Agent环境没有直接互传工具，未满足一次直接交叉回应。
尚未执行服务器micro-batch或正式RUN，当前没有U/S/H/ZS结果。

预注册判断：Full必须同时高于正式FRAMEWORK-V5 `H=81.06877662507551`和同seed、同28,228步
训练的matched online-V5 control最佳H；同一Full
checkpoint下S-off、V-off、I-off各自必须使H至少降低`1.0pp`，且`|U-S|<8`。任一失败即drop，
不启动Top-K、ridge、scale、gamma、seed或checkpoint补救搜索。

固定披露：official test用于整模型checkpoint选择和本候选配置确认；
`unseen_images_used_for_gradient=false`，`strict_blind_claim=false`。
