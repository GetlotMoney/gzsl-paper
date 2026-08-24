# V2-INNOVATION-096 结果

状态：`planned`。

研究问题：S-EDPS只用缺失证据CE训练；新增完整证据与缺失证据pair修正的一致性loss，能否进一步提高稳定性和H。

唯一改动：在相同SNPS阶段一权重、相同循环证据屏蔽和`lr=1e-4`上，新增`0.1 × MSE(correction_masked, correction_full)`。seed5须超过S-EDPS同seed`78.572828%`才追加seed7。
