# GZSL 创意树

## 根目标

```yaml
objective_id: V2-H-PLUS-3PP
framework: FRAMEWORK-V2
framework_selection: owner_selected_2026-08-22
baseline_code_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
comparison_framework: FRAMEWORK-V1
comparison_H: 74.2468
evaluation_protocol: test_selected_inductive_gzsl
primary_metric: H
target_improvement_percentage_points: 3.00
target_supported_innovations: 3
paper_method_requirement: one_coherent_framework
naming_requirement: one_method_name_plus_three_consistent_subnames
baseline_status: completed_single_seed
baseline_H: 74.023182
target_H: 77.023182
```

这里的`3.00`表示H提高3.00个百分点，不是相对增长3%。当前论文主框架是V2，正式基线为`74.023182%`，目标为`H >= 77.023182%`。V1的`74.2468%`只保留为独立比较参考。

## 当前树

```text
V2-H-PLUS-3PP
└─ baseline_completed
   ├─ V2-CONFIRM-001 / RUN-001：H=74.023182%
   └─ paper_method_goal
      ├─ shared_research_question：待证据确定
      ├─ innovation_slot_1：IDEA-001 / TG-VPR-H1（supported，paper_core_innovation）
      ├─ rejected_branch：IDEA-002 / ELPT训练式迁移（rejected，已用尽3次补救）
      ├─ rejected_branch：IDEA-003 / ICGR图像条件三组路由（rejected，两次适用补救均无提升）
      ├─ rejected_branch：IDEA-004 / ACGR全类中心化三组路由（rejected，保守补救仍无提升）
      ├─ innovation_slot_2：IDEA-005 / TST切空间语义迁移（supported，paper_core_innovation）
      ├─ rejected_branch：IDEA-006 / EPC分折先验校准（rejected，折内边际不迁移）
      ├─ rejected_branch：IDEA-007 / CATA中心对齐切空间适配（rejected，3次补救用尽）
      ├─ rejected_branch：IDEA-008 / SPA seen原型视觉锚定（rejected，破坏联合竞争）
      ├─ rejected_branch：IDEA-009 / PURL pseudo-unseen风险重加权（rejected，持续过度纠偏）
      ├─ auxiliary_branch：IDEA-010 / NTR邻域感知切空间路由（revised，当前最高观察，非核心创新）
      ├─ rejected_branch：IDEA-011 / BMR双层元路由（rejected，4个方法条件用尽）
      ├─ rejected_branch：IDEA-012 / DPT分布式原型置信度（rejected，文本不确定性无正增益）
      ├─ rejected_branch：IDEA-013 / SGT语义图残差迁移（rejected，视觉残差不可图传播）
      ├─ rejected_branch：IDEA-014 / MPR多角色原型判别（rejected，角色证据无可靠增益）
      ├─ rejected_branch：IDEA-015 / PGO梯度冲突优化（rejected，投影仅改变U/S权衡）
      ├─ reliability_branch：TG-VPR/TST/NTR新增seed9（下一RUN）
      ├─ rejected_branch：IDEA-016 / SVPG语义到视觉原型生成（rejected，seen视觉映射存在unseen域偏置）
      ├─ rejected_branch：IDEA-017 / ORT正交残差迁移（rejected，子空间与补空间均无增益）
      ├─ innovation_slot_3：IDEA-018 / CCGR类别条件几何生成（supported，paper_core_innovation）
      ├─ rejected_branch：IDEA-019 / FVRA视觉特征残差适配（rejected，seen视觉适配系统性伤害U）
      ├─ rejected_branch：IDEA-020 / EDC样本条件联合竞争（rejected，样本margin仅改变U/S权衡）
      ├─ target_78_branch：IDEA-021 / DALN原型密度感知归一化（testing）
      ├─ integration_gate：三者必须形成连续或互补逻辑
      ├─ naming_gate：一个总方法名 + 三个统一风格子名称
      └─ metric_target：四seed mean已通过；seed7仍差0.038637个百分点
```

## 节点规则

- 新节点使用 `IDEA-xxx`，并链接 `ideas/IDEA-xxx_<slug>.md`。
- 每个 Idea 必须引用本仓库的 `PAPER-xxx-Cxx` 证据，或明确标记为尚待论文证据支持的 `proposed`。
- 每个正式 Idea 只包含一个相对基线的核心改动，并预先写明成立条件和失败条件。
- 每个候选必须说明它在三创新论文主线中的角色、与其他候选的接口关系和命名候选。
- 只有获得证据与实验支持且通过整体逻辑、接口和命名检查的节点，才能标记为 `paper_core_innovation`；最终只选择三个。
- 三个最终节点必须共享一个核心研究问题，并能在同一 HTML 框架图中形成自然的数据流或机制关系；互不相关的模块不得强行并列。
- 实验开始后链接准确 Experiment 路径；结束后回填真实结果和 Idea 状态。
- 失败节点保留在树中，不删除、不重编号，也不事后修改成功门槛。

当前创新1闭环：`PAPER-001 → IDEA-001 → legacy H1 evidence → FRAMEWORK-V2 → V2-CONFIRM-001`。owner已授权直接迁移H1旧实验，IDEA-001现标记为`supported / paper_core_innovation`。
