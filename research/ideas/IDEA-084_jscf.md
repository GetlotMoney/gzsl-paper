# IDEA-084：Joint Semantic Coordination Fine-Tuning

status: testing
originality_claim: false
problem: SDRS、SEBC和SDCR按顺序独立训练，各自固定父模块；它们共同改变U/S竞争与语义残差，但从未在同一个loss下协调。
hypothesis: 冻结TG-VPR/TST-NTR/CCGR主网络，只联合微调SDRS斜率、SEBC偏置和SDCR八句残差共10个参数，可消除顺序训练冲突并超过SDCR。
evidence_refs: IDEA-046/047/075分别单独成立；IDEA-082/083表明继续增加表示变换会产生seen域偏置，因此改为只协调已成立的低维参数。
base_commit: dce549ffbb507eefd375c1b324403854cbcc5561
core_change: 推理公式不变；把三个后级模块的10个参数放入同一个小学习率seen CE阶段，并保留SDCR dropout与KL。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，句权重不塌缩，三个参数组均有梯度。
failure_condition: H不超过SDCR、任一参数饱和或联合训练只改变U/S权衡。
experiment: V2-INNOVATION-050
interim_result: RUN-001 best退回父模型；SEBC gamma被seen CE持续推高并伤害H，进入冻结SEBC的9参数RESCUE-1。
rescue_1_result: 冻结SEBC后仍退回父模型，SDRS delta持续下降；最终补救冻结SDRS，只训练SDCR八维权重。
