# IDEA-034：CCRA类别条件属性融合

```yaml
idea_id: IDEA-034
source_type: supported_CRA_refinement
evidence_refs: [V2-INNOVATION-005, V2-TRY-104]
base_commit: 6804c9c046c9487b006f585257119014baa88d0a
problem: CRA对所有类别使用同一beta，但显式属性在不同类别上的可判别性不同。
hypothesis: 由200类属性的16维PCA因子预测类别级beta残差，可增强可靠属性类别并抑制含糊类别，提高H且保持U/S平衡。
core_change: 冻结CCGR、类别中心ridge和CRA全局beta，只训练16→16→1 Gate输出每类+-4 beta残差；末层零初始化。
success_condition: seed17 H超过79.448210%，U/S任一下降不超过2个百分点，类别残差std>0.01。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过CRA父条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-104 / TG-VPR + TST + NTR + CCGR + CRA
current_attempt: V2-TRY-108
last_attempt: none
last_decision: none
```

CCRA的属性PCA允许使用全部类别属性，Gate梯度只来自seen图像；true-unseen图像不进入训练。该方向只继续优化辅助属性路径。
