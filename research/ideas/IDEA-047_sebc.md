# IDEA-047：Seen-Episodic Bias Calibration

status: supported
problem: 当前无专家最高条件S=80.904585而U=73.985535，200类联合竞争仍偏向seen类。
hypothesis: 用三个只训练100类的class-exclusive父模型，在150个seen类内部轮流模拟50类pseudo-unseen，只训练一个seen竞争扣减gamma，可把训练得到的去偏置强度迁移到真实GZSL并提高H。
evidence_refs: CONFIRM-007提供真实class-exclusive fold父模型；IDEA-046显示当前剩余误差主要表现为S/U失衡；机制复用本项目IDEA-035的EBC，不作为首次校准claim。
base_commit: 3831d795a75efaedc5cade4830ffa0e2862c0eb0
core_change: 冻结SDRS父模型和三个fold父模型，只用全局seen图像的pseudo-unseen episode训练一个gamma；不使用人工属性或真实unseen图像梯度。
success_condition: H大于77.290521，U和S任一项下降不超过2个百分点，gamma不饱和。
failure_condition: H不提高，或gamma达到98%上限。
experiment: V2-INNOVATION-013
result: RUN-002达到U/S/H/ZS=75.772560/79.346550/77.518382/83.061785%，相对SDRS父模型H提高0.227861；复用EBC机制作为无专家辅助组合成立，不作为新颖性claim。
