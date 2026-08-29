# IDEA-162：Learnable Concept Readout Probe

idea_id: IDEA-162
source_type: experiment_result + first_principles + owner_hypothesis
status: supported
problem_category: visual_grounding
mechanism_tags: [natural_prompt, shared_cross_attention, class_disjoint_probe, weak_concept_labels]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs:
  - IDEA-160
  - IDEA-161
problem: 既有诊断只用去类别名裸短语的固定CLIP方向与patch最大余弦，失败可能来自文本提示不自然、视觉/文本空间需要共享学习型对齐、或模型只能记住seen类别；尚不能判断冻结patch中是否含有可迁移的细粒度信息。
hypothesis: 使用类别无关自然鸟类提示后，再仅凭100个pseudo-seen类别的类级概念弱标签训练一套跨概念、跨类别共享的低秩文本—patch注意力读取器，应能在完全隔离的50个pseudo-unseen类别上显著提高概念AUC；若提升只出现在训练类别或打乱标签对照同样提升，则假设不成立。
core_change: 三步诊断：(1)把裸短语放入三个固定、无类别名的自然鸟类模板并重算零训练AUC；(2)冻结CLIP与全部patch/text资产，只训练共享rank-64视觉/文本残差投影和soft patch attention；(3)固定seed 7把150个formal-seen类分为100类训练、50类隔离评估，并平行训练打乱标签对照。
old_signal_or_primitive: 固定短语CLIP向量与最终层patch做无训练最大余弦，类别仍由全局点原型判断。
new_signal_or_primitive: 由类级角色描述产生的概念弱监督学习跨类别共享的概念读取规则；若成立，类别可由可复用视觉概念证据集合表示。
paradigm_shift: 本oracle检验从“类别标签学习点原型”转向“概念级弱监督学习可组合视觉原语”是否有真实信息基础；自然prompt和cross-attention本身不作为创新。
why_not_module: 共享读取器只是信息探针，不进入V4 logits、不报告U/S/H、不作为论文模块；只有class-disjoint结果证明概念原语可迁移后，才允许owner重新评估范式准入。
closest_paradigm_work: 该候选仍处于信息可读性诊断，尚未形成论文新颖性claim；若通过，正式Idea必须重新检索属性引导attention、组合式ZSL和dense vision-language grounding的近期原始论文。
minimal_falsification: 固定31个mutual-5NN概念簇、正式最终层576-patch和text-v2短语；自然模板固定为`a photo of a bird with {phrase}`、`a close-up photo of a bird showing {phrase}`、`a bird whose visible features include {phrase}`。seed 7随机排列150个formal-seen类，前100类训练、后50类评估；只评价同时含至少2个训练正类和1个评估正类的概念。共享rank-64读取器固定训练1000 updates、batch 16、AdamW lr 1e-3、weight decay 1e-4，不查看评估集选checkpoint；GPU1以同配置训练标签打乱对照。
paper_level_claim: 当前无。若通过，只能声称“冻结CLIP patch包含可由seen类概念弱监督学习并迁移到class-disjoint类别的视觉信号”，不能把探针本身包装成范式创新。
evidence_refs:
  - research/ideas/IDEA-160_full_resolution_concept_grounding.md
  - research/ideas/IDEA-161_intermediate_patch_concept_signal.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1/asset_manifest.json@sha256:d096087c9bd37d90157688e21e79b8ba6a61f0ea9b1fa91f4f544f8bc1dd1ad0
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-162-learnable-concept-readout-seed7/result.json@sha256:4f73cbbd0308b9e96af1342df2f45bb2f89ed0ffb8ec1bf6001e835101b574af
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-162-learnable-concept-readout-seed7/diagnose_learnable_concept_readout.py@sha256:640332b14da4153cf312a522447b5eb973b122cef8415c21bdc102c4a4102208
success_condition: 可评估概念不少于15；真实探针pseudo-unseen median AUC≥0.65、相对同集自然prompt冻结基线median增益≥0.05、至少60%概念AUC≥0.60，且打乱标签探针median AUC≤0.55；四项同时满足才支持“信息存在但需要学习型读取”。
failure_condition: 可评估概念不足15，或真实探针任一主门槛失败，或打乱标签对照超过0.55，则当前共享读取假设拒绝；不得通过增加容量、层、prompt或查看pseudo-unseen调参继续补救。

## 2026-08-29 三步诊断结果

| 条件 | pseudo-unseen median AUC | mean AUC | AUC≥0.60 |
|---|---:|---:|---:|
| 自然prompt冻结余弦 | 0.589784 | 0.560382 | 13/27 |
| 真实标签共享读取探针 | **0.774050** | **0.752052** | **24/27** |
| 打乱标签共享读取探针 | 0.482730 | 0.490287 | 7/27 |

- 自然prompt在全部formal-seen上的median AUC为0.562150，低于IDEA-160裸短语的0.598779；文本模板本身没有补救直接读取。
- 真实探针训练集median AUC为0.983181，固定1000 updates后在50个class-disjoint pseudo-unseen类仍达到0.774050；相对同集冻结基线增益`+0.184266`。
- pseudo-unseen中24/27概念达到AUC 0.60，18/27达到0.70；打乱标签median AUC为0.482730。
- 五项预注册成功条件全部通过，判定`learnable_concept_signal_supported`：冻结CLIP最终层patch确实包含可由seen类概念弱监督学习、并迁移到隔离类别的信号；先前失败的对象是固定文本方向与最大余弦读取，而不是视觉信息本身。
- 该结果只支持信息可读性，不证明patch注意力具有正确空间定位，也没有产生GZSL U/S/H。类级弱标签可能让探针利用与概念相关的整体外观、分类群或背景；正式范式候选仍需独立验证组合表示与类别识别收益。
- 当前不登记V4 TRY、不创建Innovation分支；需owner依据范式准入明确决定是否把“文本唯一概念弱监督→可组合视觉原语”接纳为下一正式候选。

