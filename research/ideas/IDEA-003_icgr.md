# IDEA-003：ICGR图像条件三组路由

```yaml
idea_id: IDEA-003
source_type: local_model_analysis
evidence_refs:
  - model/tg_vpr_h1/module.py
  - V2-CONFIRM-001/RUN-001
base_commit: 3dc078c0d52bf358bf24a26e48346c97de9e99ca
problem: TG-VPR对所有图像固定使用local/unique/overall等权语义，无法适应不同图像可见线索不同的问题。
hypothesis: 使用冻结图像CLS训练三组动态权重，可以在不改变TG-VPR原型的前提下提高GZSL的H。
core_change: 只训练Linear(768,64)-GELU-Linear(64,3)-softmax路由器，按图像加权三组role logits。
success_condition: seed7相对V2基线DeltaH不低于0.20个百分点，U和S各自下降不超过2个百分点，三组平均权重均不低于0.05。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: testing
paper_core_innovation: false
parent_condition: FRAMEWORK-V2 / V2-CONFIRM-001 / RUN-001
current_attempt: V2-TRY-011
```

首次TRY固定只用7057张seen训练图像训练10 epoch；CLIP缓存和TG-VPR权重冻结，不增加辅助loss。official test只在训练结束后加载，仍按项目协议披露`test_used_for_selection: true`。

## V2-TRY-010结果

首次路由得到`U=72.959208%`、`S=74.843872%`、`H=73.889524%`、`ZS=81.534684%`，相对父条件`ΔH=-0.133657`。三组平均权重为`0.251101 / 0.128730 / 0.620170`，均高于0.05，因此失败模式是“非塌缩但无提升”。按预注册规则进入RESCUE-2：输入增加图像与三组语义的3个余弦分数，其余条件不变。
