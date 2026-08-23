# IDEA-058：Cross-LLM Residual Evidence

status: supported
problem: 当前父模型主要依赖GPT-5.5八角色描述与直接类名；局部patch后续变体已收口，需要检验独立LLM描述是否提供不同语义证据。
hypothesis: 冻结SEBC父模型，加入Claude描述的CLIP文本原型残差并只训练一个beta，可利用跨LLM描述差异提高H并形成与类名/patch互补的无专家证据。
evidence_refs: 本地Claude原型与GPT全局原型平均余弦仅0.868739，说明并非同一缓存；IDEA-045证明独立类名证据有效；IDEA-057说明不应再从seen视觉生成unseen表示。
base_commit: 3c1cb0b6d2e12788132bdb10337362016c6ed0ad
core_change: 在SEBC父logits后增加Claude文本原型的CLS余弦残差，只训练一个有界beta；不使用人工属性。
success_condition: H大于当前CCPE最高77.666533，U和S任一项下降不超过2个百分点，beta不饱和。
failure_condition: H不超过CCPE，或beta达到98%上限。
experiment: V2-INNOVATION-024
result: RUN-001达到U/S/H/ZS=75.997263/79.707325/77.808093/83.523118%，四项同时提高并超过CCPE；作为创新候选成立，cache provenance与新颖性仍待核查。
