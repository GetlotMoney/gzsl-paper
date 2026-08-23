# IDEA-048：Local Patch-Semantic Residual

status: testing
problem: 当前无专家路线只用整图CLS参与分类，TG-VPR前六句局部部位描述最终被压缩到全局原型，图像局部证据没有直接出口。
hypothesis: 从冻结CLIP的576个patch中选择64个离群局部块并平均，以它们匹配“局部六句中去除类名身份方向后的文本残差”，可提供独立细粒度证据并提高H或ZS。
evidence_refs: V1缓存契约证明真实patch特征已存在；IDEA-030/IDEA-014失败说明重复全局描述无效，因此本次只使用图像局部证据和正交化局部文本。
base_commit: 90a84babdf8b1e782e99165dced3ecf4d2de7279
core_change: 在冻结SEBC父logits后增加一个局部patch-文本残差分支，只训练一个有界beta；不使用人工属性。
success_condition: H大于77.518382或ZS提高至少0.20个百分点，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H和ZS均未达到成功条件，或beta达到98%上限。
experiment: V2-INNOVATION-014
