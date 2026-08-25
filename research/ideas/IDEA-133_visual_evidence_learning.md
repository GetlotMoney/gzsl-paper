# IDEA-133：角色引导局部视觉证据学习

idea_id: IDEA-133
source_type: owner_hypothesis
status: testing
evidence_refs:
  - V2-CONFIRM-011本地RGVE-off观察：CUB端到端H=76.590293，现有主链主要依赖CLS全局特征
  - model/paper_v2.py代码观察：旧RGVE只有逐patch MLP、静态组权重和单一beta，缺少空间混合与独立局部分类目标
base_commit: 781926531af0190aaa082166fecfe762412adf0e
problem: 当前无专家FRAMEWORK-V2几乎只使用CLIP CLS完成分类，局部patch没有形成能够独立辨别细粒度类别的训练式视觉表示。
hypothesis: 在冻结CLIP patch上学习空间或部件级视觉证据，并用seen-only局部分类、注意力差异与CLIP保持损失训练，可使至少一种端到端或分阶段策略相对同资产视觉off提高至少2个H百分点，且不会造成8点以上的U或S退化。
core_change: 在现有TG-VPR→TST-NTR→CCGR全局原型链之后增加一个训练式角色引导patch证据分支；预注册四种候选结构用于确定唯一最终实现。
success_condition: CUB seed7最佳策略相对配对视觉off的Delta H至少+2.0，另一策略Delta H不低于-0.3，且U/S任一项不下降8点；通过后追加seed5/8和三数据集迁移。
failure_condition: 四候选、顺序调参和最多三次方法级补救后仍无策略达到+2 H，或2/3正式seed出现8点以上U/S退化。
experiment_queue: V2-TRY-148至V2-TRY-157为初始双策略架构筛选。
paper_core_innovation: false

