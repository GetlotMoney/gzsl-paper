# V3-CONFIRM-014 结果

状态：`planned_pending_two_round_review`，尚未启动服务器RUN，暂无U/S/H/ZS。

本Experiment只回答一个控制性问题：在四个RUN都从seed7 fresh TG开始、主batch与TG学习率时间线一致、禁止加载任何CUB训练checkpoint时，GTD、MMT、BD是否仍有独立效果。

成立必须同时满足：

- 候选best Full H减TRY042 best H至少1.0；
- 候选best checkpoint的Full H减同checkpoint Module-Off H至少1.0；
- best checkpoint的`|U-S| < 8`。

两项增益均达到0.8但不足1.0仅记为weak；其他情况drop。所有U/S/H/ZS来自同一个Full-best checkpoint，另保存best-ZS观察，不跨checkpoint拼接。

本地验证：专项21 passed、1项CUDA物理卡测试本机skip；整仓545 passed、1 skipped、3 subtests passed。服务器真实资产、CUDA RNG专项、CUDA micro-batch和修复后两轮独立Agent复核均尚未执行，因此当前不得启动正式RUN。旧四个audit smoke因模块初始化推进CUDA RNG，统一标记为`invalid_rng_mismatch`，不得作为正式证据。
