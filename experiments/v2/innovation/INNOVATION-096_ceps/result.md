# V2-INNOVATION-096 结果

状态：`rescue1_planned_scale_consistency`。

研究问题：S-EDPS只用缺失证据CE训练；新增完整证据与缺失证据pair修正的一致性loss，能否进一步提高稳定性和H。

唯一改动：在相同SNPS阶段一权重、相同循环证据屏蔽和`lr=1e-4`上，新增`0.1 × MSE(correction_masked, correction_full)`。seed5须超过S-EDPS同seed`78.572828%`才追加seed7。

RUN-001得到`U/S/H/ZS=76.982599/80.230141/78.572828/84.121776%`、selected iteration=`2820`，与S-EDPS逐项相同。selector权重有约`1e-5`级变化，但训练loss仅发生约`1e-6`至`1e-5`级变化，原始一致性项量级不足以改变预测。

RESCUE-1只把一致性权重从`0.1`提高到`100`，使该项约进入CE的千分之几量级；其余条件保持不变。模型SHA256：`37da7d28fb2f14a571681efc87359888f24f8afed419c99f65e67ad231432f4d`；最后checkpoint SHA256：`01ee5127e29c1c7d50c26772488fb1ae0bc3e1ce97454937418a2d88e806d2b7`。
