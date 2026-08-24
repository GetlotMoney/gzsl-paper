# V2-INNOVATION-087 结果

状态：`rejected_below_top3`。

首次RUN在训练前因MHPS schema漏接12维特征分支失败，输出保留且不计有效实验。修复后的RUN-001-RERUN构建608个pair：304个全部top2错误+304个最低margin正确，标签严格1:1。

有效RERUN最高`U/S/H/ZS=76.843750/79.966521/78.374042/84.016234%`，selected iteration=`141`，低于稳定SNPS top-3 `0.092668`。确定性匹配仍因样本量下降而过拟合seen边界，IDEA-121拒绝。

模型SHA256：`d4c11858f14e6fa0dd0f7e9b5d1a314f1ff7bbaa848954db2cc8f9b19de43b79`；最后checkpoint SHA256：`ac62ce305326778b5b211315c96a912f72503e83783a91da3757cb470e8ce172`。
