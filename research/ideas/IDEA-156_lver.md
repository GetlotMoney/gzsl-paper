# IDEA-156：Local-View Evidence Routing

idea_id: IDEA-156
source_type: experiment_result + code_analysis + first_principles
status: testing
evidence_refs:
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v3/fresh-effective/V3-TRY-043/metrics.json
  - 645b609:experiments/v3/confirmation/CONFIRM-014_fresh_effective_modules/result.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/assets/v3/CUB_openai_vitl14_336_lver_4view_v2/asset_manifest.json
base_commit: bb7d900910ef317142e956537d2d84a2b074f9d8
problem: GTD已改善true-unseen原型的跨组位置，但其best checkpoint仍有18.071478%的seen→unseen、8.205715%的unseen→seen和12.035393%的unseen组内错误；高频错误集中在Vireo、Flycatcher、Tern、Warbler等细粒度近邻。冻结最终CLS把局部差异压成单向量，而历史内部patch直接文本匹配没有稳定净纠错方向。
hypothesis: 将四个固定75%角裁剪分别重新经过完整冻结CLIP图像编码器，并只在TG+GTD低margin Top-3内用共享路由形成零和局部残差，可以获得比内部patch更可靠的文本对齐局部证据，并同时通过相对匹配父条件与同checkpoint关闭的1H门槛。
core_change: 在TG+GTD 200类logits后增加四局部CLS共享路由；候选内残差中心化，非候选不变，强度从0初始化；不移动原型，不使用框、部位、属性或unseen图像梯度。
success_condition: 相对V3-TRY-046 best H至少+1.0；同checkpoint Full H减LVER-Off H至少+1.0；best checkpoint的|U-S|<8。
failure_condition: 任一独立增益低于0.8或|U-S|>=8即drop；两项均至少0.8但不足1.0仅记weak。
experiment: experiments/v3/innovation/INNOVATION-015_lver/
