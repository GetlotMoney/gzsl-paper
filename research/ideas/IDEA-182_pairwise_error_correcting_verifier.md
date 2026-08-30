---
idea_id: IDEA-182
source_type: experiment_result
evidence_refs:
  - IDEA-182 class-disjoint Gate @ 6043e1d: Parent 67.9755%, PECV 72.5250%, gain +4.5495pp, output /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-182-pecv-gate-seed7
  - FRAMEWORK-V4 @ 52088f69d7ac4e574e7b63c28b21ac0da7789933
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
problem: TG+GTD通常召回真类但相近类别最终排序仍错误。
hypothesis: 共享候选两两验证器从seen类学习角色差异判别后，可以在正式200类联合竞争中提高H。
core_change: 在TG+GTD Top-5内加入反对称零和PECV纠错，并从update 1与TG、GTD同步训练。
success_condition: 相对匹配Parent条件与同checkpoint关闭PECV均提高至少1.0 H，且U/S差小于8点。
failure_condition: 任一ΔH不足1.0，或U/S差达到8点。
status: testing_formal
---

# IDEA-182：PECV正式验证

Gate只证明了跨类别排序信号存在。本实验不加载Gate、TG或GTD checkpoint；TG、GTD、PECV使用同一seed，从update 1同步训练200名义epoch。
