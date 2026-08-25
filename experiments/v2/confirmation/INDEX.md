# V2 Confirmation

| Experiment | 目的 | 状态 | 目录 |
|---|---|---|---|
| V2-CONFIRM-001 | FRAMEWORK-V2首个新仓库正式CUB基线 | completed | `CONFIRM-001_v2_baseline/` |
| V2-CONFIRM-002 | 无三折、无阶段冻结、固定第50轮的统一seen训练 | completed; seen-biased gain | `CONFIRM-002_unified_seen_training/` |
| V2-CONFIRM-003 | validation冻结后的无专家/专家trainval最终评估 | completed; expert H 78.751611 | `CONFIRM-003_standard_clip_final/` |
| V2-CONFIRM-004 | Chen-style端到端无专家/专家整模型选模 | completed; expert H 78.134714 | `CONFIRM-004_chen_style_end_to_end/` |
| V2-CONFIRM-005 | Chen-style固定边界分阶段无专家整模型选模 | closed; best H 76.006848 | `CONFIRM-005_chen_style_stagewise/` |
| V2-CONFIRM-006 | Chen-style分阶段pseudo-unseen辅助目标 | rejected; H 75.948676 | `CONFIRM-006_chen_stagewise_pseudo_unseen/` |
| V2-CONFIRM-007 | Chen-style真正class-exclusive三fold共享迁移 | closed; best H 75.849154 | `CONFIRM-007_chen_class_exclusive/` |
| V2-CONFIRM-008 | 三折validation冻结后的完整150类最终评估 | completed; H 74.814246 | `CONFIRM-008_threefold_frozen_final/` |
| V2-CONFIRM-009 | 三数据集可追溯Pure CLIP与Mean8基线 | planned | `CONFIRM-009_multidataset_traceable_baselines/` |
| V2-CONFIRM-010 | 三数据集text-v2的B0/B1/M3 seed7恢复确认 | planned | `CONFIRM-010_text_v2_recovery/` |
