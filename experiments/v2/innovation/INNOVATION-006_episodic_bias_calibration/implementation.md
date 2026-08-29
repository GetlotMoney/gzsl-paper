# 实现说明

每个fold以100类为pseudo-seen、50类为pseudo-unseen，CRA ridge仅用100个pseudo-seen视觉中心拟合；gamma用32/32图像联合CE训练。正式推理只对150个真实seen类别logit扣减gamma，ZS的50类竞争不变。
