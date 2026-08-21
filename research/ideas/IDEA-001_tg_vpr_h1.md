# IDEA-001：TG-VPR-H1结构化语义重参数化

```yaml
idea_id: IDEA-001
status: testing
evidence_refs:
  - PAPER-001
base_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
framework: FRAMEWORK-V2
paper_role: innovation_slot_1_candidate
paper_core_innovation: false
method_name_candidate: TG-VPR
experiment_ref: pending_control_experiment
framework_baseline_ref: V2-CONFIRM-001/RUN-001
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
