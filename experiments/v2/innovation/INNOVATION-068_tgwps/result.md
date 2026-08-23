# V2-INNOVATION-068 结果

状态：`planned`。尚无正式结果。

首次RUN虽完成，但pair_dataset仅169个，与配置声明的`all_same_group_top2_soft_gate`不一致。原因是text-only schema漏入hard-margin分支；该结果标记`invalid_contract`，不计方法结论和预算，原输出保留。修复后使用全新`RUN-001-RERUN-001`目录执行同一配置。
