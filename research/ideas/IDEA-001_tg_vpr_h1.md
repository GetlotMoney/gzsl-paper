# IDEA-001：TG-VPR-H1结构化语义重参数化

```yaml
idea_id: IDEA-001
status: supported
source_type: paper
problem_category: semantic_representation
mechanism_tags: [role_grouping, prototype_reparameterization, topology_preservation]
evidence_refs:
  - PAPER-001
  - PAPER-004
  - PAPER-009
  - PAPER-010
base_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
framework: FRAMEWORK-V2
paper_role: innovation_slot_1_candidate
paper_core_innovation: true
novelty_claim: fixed_role_group_reparameterization_not_first_GPT_description_prompting
method_name_candidate: TG-VPR
experiment_ref: experiments/v2/evidence/legacy_h1
framework_baseline_ref: V2-CONFIRM-001/RUN-001
legacy_evidence_refs:
  - V5-INNOVATION-024
  - V5-ABLATION-014
  - V5-TUNE-005
  - V5-TUNE-006
```

## 研究问题

相比直接对GPT视觉描述做自由self-attention聚合，先按local、unique、overall组织语义，再进行共享Value重参数化，能否得到更稳定、可解释的类别原型？

## 唯一核心改动

将多句视觉描述改造成三组固定等权语义，经共享Value路径和topology约束形成seen类原型；unseen类保留Mean8。

## 可证伪假设

在相同数据、seed和评估协议下，完整TG-VPR-H1应优于关闭Value重参数化的baseline-off条件，并且提升不能只来自test搜索。

## 成立与失败

- 成立：正式控制实验中完整条件的H高于baseline-off，且U/S没有单边崩塌。
- 失败：完整条件不优于baseline-off，或收益无法在预注册重复中出现。

## 论文衔接

创新1输出结构化类别原型；后续创新2、创新3只能围绕这些原型的unseen迁移和图像条件证据选择继续，不能另起无关模块。

## 已迁入支持结果

- Value路径相对无Value：`+6.376252 H`；
- 三组结构相对单组Value：`+0.131688 H`；
- 可学习组权重相对固定`1/3`：`-0.003517 H`，因此删除；
- 最终四seed H mean=`73.853094`，range=`0.313729`；
- 当前仓库V2正式seed 7基线H=`74.023182`，与旧固定等权seed 7逐值一致。

结论：IDEA-001获得机制消融、多seed和当前仓库基线支持，标记为论文核心创新1。旧证据是test-exposed、非blind-test，不扩展为独立confirmation结论。
