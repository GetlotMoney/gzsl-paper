# 实现说明

ARA先用7057张seen训练图像闭式拟合`CLIP(768)→attribute(312)`的ridge映射，再冻结映射，仅训练一个有界融合系数beta。推理logit为CCGR主语义logit与属性余弦logit之和。beta为0时严格关闭ARA；最终组合不使用SDM。
