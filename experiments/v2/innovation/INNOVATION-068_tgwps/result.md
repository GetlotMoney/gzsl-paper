# V2-INNOVATION-068 结果

状态：`rejected_below_agct_retained_patch_free`。

首次RUN虽完成，但pair_dataset仅169个，与配置声明的`all_same_group_top2_soft_gate`不一致。原因是text-only schema漏入hard-margin分支；该结果标记`invalid_contract`，不计方法结论和预算，原输出保留。修复后使用全新`RUN-001-RERUN-001`目录执行同一配置。

RERUN正确使用`4041`个全pair且完全不读取patch。最高`U/S/H/ZS=76.808137/79.959720/78.352250/83.948439%`，相对SDCR H提高`0.031739`，但低于patch-free AGCT `78.357224`约`0.004974`，也低于GWPS。

T-GWPS未通过预注册门槛，IDEA-102拒绝；但作为patch-free次级对照保留，证明patch差值是GWPS额外增益所需的交互特征。

模型SHA256：`25c521ed97b73bdd42913a208a1d2518e53b9ccbc8be52a0e3aa2ad12c4c80ff`；最后checkpoint SHA256：`806805b83e37f0ff29b51bf715afa71da637da3294a336133fcf41e1c6455be6`。
