# V2-INNOVATION-099 结果

状态：`rejected_train_unseen_sign_reversal`。

审计显示“文本距离最大角色”的方向在unseen可达错误上正确125/288，其中71个是S-EDPS方向错而该角色方向对；同时有476个正确样本可由它修复S-EDPS风险，具备互补性。

唯一改动：冻结S-EDPS 12维selector，只新增并训练一个top-discriminative role差值系数。关闭时逐位复现S-EDPS；seed5须超过`78.572828%`才追加seed7。

RUN-001完整训练后best-H严格保持父模型`U/S/H/ZS=76.982599/80.230141/78.572828/84.121776%`，selected iteration=`-1`。独立best-ZS同样为`84.121776%`、iteration=`-1`，没有跨checkpoint拼接。

训练集4458个pair中top2错误仅112/304被该角色给出正确方向，最后role_weight学成`-0.202035`；official unseen审计则希望正方向，构成明确train/unseen符号翻转。IDEA-132拒绝，不做非负投影或幅度补救。

本实验父S-EDPS和当前阶段均使用official test选择，明确标记`nested_official_test_selection: true`。模型SHA256：`d120b6d34a76dd86b4f22e3613852197202efcb414740b1557d930376beb6ece`；最后checkpoint SHA256：`f19f0e76228f891c19986e87680d45cc960613966c944b1c3d50fcb3e10a078b`。
