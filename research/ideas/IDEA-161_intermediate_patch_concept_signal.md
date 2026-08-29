# IDEA-161：Intermediate-Patch Concept Signal

idea_id: IDEA-161
source_type: experiment_result + first_principles
status: revised
problem_category: visual_grounding
mechanism_tags: [intermediate_tokens, concept_grounding, frozen_clip, prequeue_oracle]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs:
  - IDEA-160
problem: IDEA-160证明最终层576-patch比36-patch保留更多概念信息，但仍未通过门槛；尚不能区分细粒度信号在更早Transformer层存在但在最终层被全局语义冲淡，还是冻结CLIP各层都不足以稳定支撑这些文本概念。
hypothesis: 若细粒度角色概念主要存在于冻结CLIP的中间局部token，保持同一31概念和AUC口径时，layer 12/16/20中至少一层应通过原概念门槛，并相对同一1000张样本的正式最终层576缓存取得至少0.03中位AUC增益。
core_change: 从CUB seen训练集按150类确定性分层抽取1000张图像，在两张GPU上重新前向OpenAI CLIP ViT-L/14@336，读取第12/16/20/24个Transformer block的576个局部token；各层经原ln_post与visual.proj映射后，使用同一31个概念的最大patch余弦计算AUC。无训练、无人工属性/部位/框、无true-unseen图像。
old_signal_or_primitive: IDEA-160只使用冻结CLIP最终第24层的576个局部token。
new_signal_or_primitive: 把类别概念落地依赖的视觉原语改为中间层局部token，检验细节是否在深层全局化前存在。
paradigm_shift: 若成立，后续视觉表示将基于层级局部概念证据，而不是只读取最终层全局化token；本次oracle本身只是范式前提诊断，不作为创新。
why_not_module: 该实验不增加网络、Gate、Head、Top-K、校准或loss，只读取冻结CLIP既有中间状态并检验信息是否存在。
closest_paradigm_work: 候选在pre-queue最小证伪阶段失败，未启动近期相关工作系统检索，也不形成多层视觉创新claim。
minimal_falsification: 固定seed 7，从每个seen类抽6或7张图共1000张；固定layers=12/16/20/24、mutual-5NN threshold=0.85形成的31个概念、最大patch余弦和类级AUC。任一中间层须同时满足median AUC≥0.60、至少60%概念AUC≥0.60、相对同样本正式最终层缓存median AUC增益≥0.03。
paper_level_claim: 当前无；结果只否定固定CLIP文本方向对中间层token的直接读取，不证明中间层不含可由学习型读取器提取的细粒度信息。
evidence_refs:
  - research/ideas/IDEA-160_full_resolution_concept_grounding.md
  - experiments/v3/PATCH_ASSET_REBUILD_AUDIT.md
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-161-intermediate-patch-concepts-seed7/result.json@sha256:574986e1f671648e6c1b043cca401f5acfc826d57ad3e44bc3f05024866f8402
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-161-intermediate-patch-concepts-seed7/diagnose_intermediate_patch_concepts.py@sha256:d2bddd4a34fb0766bf2a48e04d39a61b64c11a431c1b6f828d7ef7c562b495ef
success_condition: 任一layer 12/16/20同时通过median AUC、概念覆盖比例和相对最终层增益三项门槛，才允许提出新的层级视觉表示Idea。
failure_condition: 没有任何中间层同时通过三项门槛，则拒绝“细粒度信号藏在当前冻结CLIP中间层”的假设，不生成全量中间层资产、不进入Innovation或训练。

## 2026-08-29 双卡1000图最小证伪结果

| 表示 | median AUC | mean AUC | AUC≥0.60 |
|---|---:|---:|---:|
| layer 12 | 0.506667 | 0.507697 | 4/31 |
| layer 16 | 0.513308 | 0.526411 | 5/31 |
| layer 20 | 0.571931 | 0.583142 | 13/31 |
| layer 24重新提取 | 0.593183 | 0.607117 | 14/31 |
| 正式最终层576缓存，同1000图 | 0.595085 | 0.606672 | 14/31 |
| 标签打乱对照 | 0.497674 | 0.503894 | 1/31 |

- layer 24与正式最终层缓存的AUC几乎一致，说明原图顺序、预处理、文本概念和中间层读取链没有发生口径漂移。
- layer 12/16接近随机，layer 20虽恢复部分语义，但仍低于最终层；没有中间层通过任一完整成功条件。
- 原始判定“中间层不含细粒度信息”证据越界，2026-08-29修订为`revised`：本实验只证明固定短语向量经最终`ln_post+visual.proj`后做最大余弦不能直接读取中间层；它没有训练共享投影或cross-attention，不能据此否定信息存在。
- 真实边界：该结论针对OpenAI CLIP ViT-L/14@336的layers 12/16/20/24经最终ln_post+visual.proj后的局部token与当前31个类级弱监督概念；不能外推为所有视觉编码器或所有局部监督都无效。
