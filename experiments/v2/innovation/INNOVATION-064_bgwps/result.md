# V2-INNOVATION-064 结果

状态：`rejected_overbalanced_pairs`。

逆频率权重把top1/top2类别系数设为`0.536653/7.320652`，组合pair权重std达到`5.399093`。所有训练条件显著降低H并最终稳定约`76.75%`，best严格退回父模型`H=78.320510%`与零selector。

完全类别平衡过度放大少数top2样本，IDEA-098拒绝。下一独立Experiment使用平方根逆频率，把相对补偿从约13.6倍降到约3.7倍。

模型SHA256：`5a3481831df9eda7fece6a776ff4f9c1f588d93b9c2aa0dd6012cc9cad890258`；最后checkpoint SHA256：`21ab896b8a6ac4535d10929a3485bee3460c9d22864439cb082ace5dc14001e2`。
