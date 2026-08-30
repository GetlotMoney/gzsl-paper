---
idea_id: IDEA-182
source_type: experiment_result
evidence_refs:
  - HCVC Gate 0 Parent: macro Top-1 67.9755%, frozen Top-5 coverage 95.35%, output /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-171-hcvc-gate-seed7
  - FRAMEWORK-V4 @ 52088f69d7ac4e574e7b63c28b21ac0da7789933
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
problem: Parent通常已把真类放入Top-5，但相近类别的最终排序仍错误；继续生成视觉特征已经被HCVC否定。
hypothesis: 一个不接收类别ID、只读取冻结图像特征与两候选角色文本的共享裁判，可以把100类学到的“候选差异如何判别”迁移到完全隔离的50类，并改善Parent Top-5内部排序。
core_change: 在冻结Parent Top-5内增加反对称、零和的候选两两纠错分数；不改TG、GTD、图像特征或候选集合。
success_condition: 50类class-disjoint宏平均Top-1相对Parent提高至少1.0个百分点，打乱候选语义后至少损失0.5个百分点，且关闭模块逐值返回Parent。
failure_condition: 未达到任一成功条件，或纠错损坏数不低于纠正数。
status: testing
---

# IDEA-182：候选两两纠错验证器（PECV）

一句话：Parent负责召回，PECV只回答“这两个相似候选中，当前图像更支持谁”。

它不是新的视觉生成器，也不读取unseen图像梯度。训练只使用100个dev-seen类的冻结特征与错误候选；2355张dev-unseen图像只在训练结束后评估。

当前阶段只验证可迁移的排序增益，不提前宣称范式级创新或正式GZSL增益。
