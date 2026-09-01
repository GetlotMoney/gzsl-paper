# V6-INNOVATION-001 / V6-TRY-006 result

状态：`planned_pre_run`。尚未执行服务器micro-batch或正式RUN，当前没有U/S/H/ZS结果。

预注册判断：Full必须高于准确父条件FRAMEWORK-V5 `H=81.06877662507551`；同一Full
checkpoint下S-off、V-off、I-off各自必须使H至少降低`1.0pp`，且`|U-S|<8`。任一失败即drop，
不启动Top-K、ridge、scale、gamma、seed或checkpoint补救搜索。

固定披露：official test用于整模型checkpoint选择和本候选配置确认；
`unseen_images_used_for_gradient=false`，`strict_blind_claim=false`。
