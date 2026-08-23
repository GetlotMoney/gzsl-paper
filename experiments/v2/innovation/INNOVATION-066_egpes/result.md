# V2-INNOVATION-066 结果

状态：`rejected_below_gwps`。

50分位硬pair训练集包含`386`个样本，top1真类比例=`0.621762`，明显缓解GPES的169 pair小样本问题。最高`U/S/H/ZS=76.747566/80.057371/78.367537/83.954543%`，高于SDCR与AGCT，但低于GWPS `78.414246%`。

169/386/4041三种pair规模均已覆盖，soft-gate全pair GWPS最优。IDEA-100拒绝为低于GWPS，并关闭pair范围/标签权重轴。

模型SHA256：`e85068749361ac1b7c075263c8ecb3088dddce7e79d86fd9e4ca4adc6e84f060`；最后checkpoint SHA256：`8ba318427a73ce2f815a33afd31af0fa0530f73169c3aa02de75177c1d1908df`。
