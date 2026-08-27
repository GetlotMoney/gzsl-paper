# IDEA-134：稀疏局部证据

idea_id: IDEA-134
source_type: owner_and_code_hypothesis
status: rejected
base_commit: 3be6cb23ac8d19af909a84c5403496a961e6b531
implementation_commit: 61d9e71131e0d09359038988e529483142fd4a2b
problem: 旧Spatial-RGVE对576个patch、200类和3组查询计算全量注意力，时间和显存高；其一段式累计与单移除贡献都只有+0.558258 H，未达到owner固定的+1硬门槛。
hypothesis: 每个4x4空间区域保留一个与冻结CLS最相似的patch，用共享64维投影形成独立全类别局部logit，并以零初始化图像门控融合，可在显著降低patch复杂度的同时，相对TG父条件提高至少1个H百分点。
inputs: [冻结CLIP CLS, 每图16个确定性区域patch, 八角色文本形成的3组类别查询]
output: 独立local logits与TG logits的图像条件融合
human_annotations_used: false
training_strategy: end_to_end_joint_200
parent_metrics: {U: 78.407878, S: 74.983871, H: 76.657659, ZS: 86.146760}
success_condition: seed7的TG+SLE H至少77.657659且U/S差小于8点；因为关闭SLE严格回到TG，此差值同时作为累计加入与单模块移除贡献。
failure_condition: H低于77.657659或U/S差达到8点时直接拒绝当前SLE结构，不以运行优化掩盖方法失败。
novelty_claim: pending_related_work_search_after_empirical_pass
result: V3-TRY-001完成200名义epoch和201次official评估；U/S/H/ZS=59.506238/88.204509/71.067523/83.791411，best epoch=62，visual beta mean=0.299737。
delta_vs_tg: {U: -18.901640, S: +13.220638, H: -5.590136, ZS: -2.355349}
decision: drop。16-patch资产将patch体积从10.43GB降到0.29GB，batch50峰值allocated约271.6MB、总GPU显存约1.79GB，运行优化成立；但独立local CE造成严重seen偏置且beta饱和，U/S差28.698272达到淘汰线。按owner硬标准不做参数补救，不晋级正式FRAMEWORK-V3。
