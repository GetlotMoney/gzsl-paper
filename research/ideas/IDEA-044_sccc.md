# IDEA-044：Sample-Conditioned Competition Calibration

status: rejected
problem: 当前最佳无专家父模型仍存在seen/unseen样本级竞争偏差，H=76.006848。
hypothesis: 使用父logits的seen/unseen置信度差预测每张图像独立的seen扣减量，可在不改变原型的情况下提高U/S调和均衡。
base_commit: 626a94c1afff7a60eca3de4b8227110d48ffbbdc
core_change: 冻结父模型，只训练零初始化6维置信度gate；seen内部三fold模拟竞争分区。
success_condition: H超过76.006848，最终目标H大于等于77.023182；gamma不饱和。
failure_condition: best H不超过父模型，或gamma达到98%上限。
experiment: V2-INNOVATION-010
