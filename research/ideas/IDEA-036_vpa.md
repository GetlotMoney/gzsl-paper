# IDEA-036：VPA属性视觉原型残差

```yaml
idea_id: IDEA-036
source_type: reverse_attribute_mapping
evidence_refs: [PAPER-002, V2-INNOVATION-005]
base_commit: e3f03a880fdc051c8e71d087be33e87188b9069e
problem: CRA通过图像→属性映射改善显式语义，但属性→视觉中心的反向映射可能提供独立的unseen视觉原型证据。
hypothesis: 用seen属性到视觉中心的ridge生成200类视觉原型，并以训练式beta融合到CRA，可提高ZS和H。
core_change: 在冻结CRA上增加属性→视觉中心ridge分支；beta=0回到CRA，正反ridge均使用0.01。
success_condition: seed17 H超过79.448210%，U/S任一下降不超过2个百分点，beta不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过CRA父条件。
status: supported
paper_core_innovation: false
parent_condition: V2-TRY-104 / TG-VPR + TST + NTR + CCGR + CRA
current_attempt: none
last_attempt: V2-TRY-124
last_decision: promote_auxiliary_after_combination_reliability
current_tune: none
```

VPA的正反ridge和beta训练都只使用seen类别中心或seen图像；true-unseen图像不进入梯度。语义→视觉ridge已有论文先例，因此只检验本框架中的组合价值。

## V2-TRY-118结果

第9轮得到`U=74.306595%`、`S=85.574788%`、`H=79.543609%`、`ZS=87.125778%`，相对CRA提高H `0.095400`和ZS `0.906229`，但U下降`1.013190`。VPA提供强类内判别、同时加重seen偏置；下一步与已支持EBC做训练式组合，再决定是否运行VPA多seed。

## 四训练seed支持结论

seed7/17/27/37的H为`79.524497/79.543609/79.463000/79.454045%`，H mean/min/max/range=`79.496287/79.454045/79.543609/0.089564`；四个seed均提高H与ZS。IDEA-036标记`supported`辅助，最终是否采用由VEBC组合可靠性决定。

seed17反向ridge `0.01/0.1/1.0`的H为`79.543609/79.674486/79.662435%`。0.1单VPA最高，0.01的ZS最高；追加一次0.1与VEBC组合后关闭该轴。
