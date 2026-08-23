# IDEA-076：Sentence-Dropout Consistency Calibration

status: testing
problem: SDCR每批mask一句的两seed增益可靠但较小；学生仅用CE训练，没有显式要求缺句预测保持完整8句语义。
hypothesis: 在SDCR上增加0.1×KL(dropout student || full-sentence teacher)，可让缺句训练更稳定并超过SDCR，同时不改变推理结构。
evidence_refs: IDEA-075 SDCR两seed可靠且mask1优于mask2；一致性教师使用同一步当前完整8句预测并停止梯度。
base_commit: 408a122284697d59643052406058790231a13001
core_change: SDCR mask1、CASR父权重和推理结构不变，只增加dropout学生到完整8句教师的一致性KL。
success_condition: H大于SDCR最高78.320510，U和S任一项下降不超过2个百分点，推理权重std/min均通过非塌缩门槛。
failure_condition: H不超过SDCR或权重塌缩。
experiment: V2-INNOVATION-042
