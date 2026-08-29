# IDEA-126：Teacher-Forced Pair Selector

status: rejected
problem: 当前pair训练只保留真实类已经位于top2的样本，真实类掉到第3名以后完全不参与selector监督。
hypothesis: 对seen错误图像使用“父top1 vs 真实类”教师pair，可覆盖更多错误模式并提高top1/top2推理selector的泛化。
evidence_refs: SNPS全量pair只含true-in-top2样本；LSCR三类CE失败但表明真类top3样本可构建；现有pair监督覆盖有限。
base_commit: 8aefc076b1342eb5642e7fa9f8103822b475b484
core_change: seen训练pair构造改为错误样本top1-vs-true、正确样本top1-vs-top2；错误权重下限0.25，推理不变。
success_condition: seed5 H大于稳定SNPS top-3 78.466710；正提升后追加seed7。
failure_condition: H不超过top-3、错误pair为空或U/S任一下降超过2个百分点。
experiment: V2-INNOVATION-092
paper_core_innovation: false
result: 4604个相关pair，比SNPS多146个教师强制错误；H=78.458154，低于top-3 0.008557。
decision: 扩大seen错误覆盖未改善unseen泛化，拒绝。
