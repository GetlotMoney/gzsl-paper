# V2-INNOVATION-054 结果

状态：`rejected_patch_weighting_no_gain`。

中心化权重满足预注册边界：mean/std/min/max=`1.000000/0.068121/0.750000/1.052839`，因此没有CE总尺度混杂。但所有训练条件仍低于父模型，best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`、selected iteration=`-1`。

PGSD与CPGSD说明：top2 patch虽在较弱SEBC父链提供局部正信号，但既不适合叠加到SDCR推理，也不适合重加权SDCR句子训练。IDEA-088拒绝并关闭当前patch结合轴。

模型SHA256：`5d555b38b26c47b9f376e68d709376c1d2615333fd7d08338113c4d63e7ca5c4`；最后checkpoint SHA256：`fd078082041e58f74ec71c84a6fe5ebbd0606e34a54286c84ac50b36d7dbee57`。
