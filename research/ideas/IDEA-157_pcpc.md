# IDEA-157：Pair-Contrast Patch Comparator

idea_id: IDEA-157
source_type: experiment_result + code_analysis + first_principles
status: rejected
evidence_refs:
  - /data/lby/projects/cv_project/GZSL_Warehouse/tries/v3/fresh-effective/V3-TRY-043/metrics.json
  - experiments/v3/PATCH_ASSET_REBUILD_AUDIT.md
  - research/ideas/IDEA-143_multiscale_candidate_difference.md
base_commit: bb7d900910ef317142e956537d2d84a2b074f9d8
problem: 旧CCPE/AGPT分别为每类寻找最高patch，两个候选可能依赖不同区域；IDEA-143冻结差向量与手工投票又整体有害。剩余近邻错误需要在同一patch上直接学习候选A相对B的可见差异，而不是继续叠加绝对类别patch分数。
hypothesis: 对TG+GTD Top-2候选的八角色文本差使用共享低秩视觉/文本投影，在同一576 patch上形成signed evidence并施加反对称零和纠错，seen hard-pair监督可迁移到unseen候选，并同时通过双1H门槛。
core_change: 只修改父Top-2 logits；同一patch上的A-B角色差证据经|e|软选择后聚合为signed delta，对两候选施加+delta/-delta；不移动原型、无类别专属参数、无seen/unseen身份参数。
success_condition: 相对V3-TRY-048 best H至少+1.0；同checkpoint Full H减PCPC-Off H至少+1.0；best checkpoint的|U-S|<8。
failure_condition: 任一独立增益低于0.8或|U-S|>=8即drop；两项均至少0.8但不足1.0仅记weak。
experiment: experiments/v3/innovation/INNOVATION-016_pcpc/
result: V3-TRY-050 best U/S/H/ZS=78.604460/77.336574/77.965362/85.342348；相对V3-TRY-048 ΔH_add=-0.190046，同checkpoint ΔH_remove=-0.190046，PCPC降低U并未形成可用局部判别增益，按预注册drop且不调参。
