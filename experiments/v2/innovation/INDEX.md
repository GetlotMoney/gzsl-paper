# V2 Innovation

| Experiment | Idea | 目的 | 状态 | 目录 |
|---|---|---|---|---|
| V2-INNOVATION-001 | IDEA-002 | 固定10%保守unseen原型迁移 | retained_test_time_observation | `INNOVATION-001_conservative_unseen_transfer/` |
| V2-INNOVATION-002 | IDEA-005 | TST切空间语义迁移 | supported | `INNOVATION-002_tangent_semantic_transport/` |
| V2-INNOVATION-003 | IDEA-018 | CCGR类别条件几何生成 | supported | `INNOVATION-003_class_conditioned_geometric_generator/` |
| V2-INNOVATION-004 | IDEA-029 | ARA显式属性残差对齐 | retained out-of-scope: expert attributes | `INNOVATION-004_attribute_residual_alignment/` |
| V2-INNOVATION-005 | IDEA-033 | CRA类别中心属性对齐 | retained out-of-scope: expert attributes | `INNOVATION-005_centroid_ridge_alignment/` |
| V2-INNOVATION-006 | IDEA-035 | EBC episodic偏置校准 | retained out-of-scope: attribute parent | `INNOVATION-006_episodic_bias_calibration/` |
| V2-INNOVATION-007 | IDEA-037 | VPA+EBC互补组合 | retained out-of-scope: expert attributes | `INNOVATION-007_visual_prototype_bias_composition/` |
| V2-INNOVATION-008 | IDEA-038 | JBEC联合双向校准 | retained out-of-scope: attribute parent | `INNOVATION-008_joint_bidirectional_episodic_calibration/` |
| V2-INNOVATION-009 | IDEA-041 | CNRA类名残差对齐 | retained out-of-scope: attribute parent | `INNOVATION-009_class_name_residual_alignment/` |
| V2-INNOVATION-010 | IDEA-044 | SCCC样本条件竞争校准 | rejected | `INNOVATION-010_sccc/` |
| V2-INNOVATION-011 | IDEA-045 | NCRA类名与GPT长描述双语义残差 | supported auxiliary（无专家H=77.201125） | `INNOVATION-011_ncra/` |
| V2-INNOVATION-012 | IDEA-046 | SDRS语义分歧驱动类名残差缩放 | supported auxiliary marginal（H=77.290521） | `INNOVATION-012_sdrs/` |
| V2-INNOVATION-013 | IDEA-047 | SEBC seen内部episode竞争去偏置 | supported auxiliary（无专家H=77.518382） | `INNOVATION-013_sebc/` |
| V2-INNOVATION-014 | IDEA-048 | LPSR局部patch-文本残差 | rejected（H仅+0.003747且方向错误） | `INNOVATION-014_lpsr/` |
| V2-INNOVATION-015 | IDEA-049 | CCPE每类独立局部patch证据 | supported candidate（当前无专家H=77.666533） | `INNOVATION-015_ccpe/` |
| V2-INNOVATION-016 | IDEA-050 | SCPE空间一致局部patch证据 | rejected（低于CCPE） | `INNOVATION-016_scpe/` |
| V2-INNOVATION-017 | IDEA-051 | MPPE六局部部位独立patch证据 | rejected（六路噪声累积） | `INNOVATION-017_mppe/` |
| V2-INNOVATION-018 | IDEA-052 | CNPE seen参考归一化patch证据 | rejected（低于CCPE，保留互补信号） | `INNOVATION-018_cnpe/` |
| V2-INNOVATION-019 | IDEA-053 | DSPE绝对与相对双尺度patch证据 | rejected（联合与分阶段均失败） | `INNOVATION-019_dspe/` |
| V2-INNOVATION-020 | IDEA-054 | PCME局部patch分数共识边际 | rejected（方向合理但无增益） | `INNOVATION-020_pcme/` |
| V2-INNOVATION-021 | IDEA-055 | ECPE episode训练的CCPE | rejected（fold方向相反） | `INNOVATION-021_ecpe/` |
| V2-INNOVATION-022 | IDEA-056 | CRPE类别语义可靠性patch缩放 | rejected（类别斜率无增益） | `INNOVATION-022_crpe/` |
| V2-INNOVATION-023 | IDEA-057 | LVPG局部视觉原型生成 | rejected（seen视觉域偏置） | `INNOVATION-023_lvpg/` |
| V2-INNOVATION-024 | IDEA-058 | CLRE跨LLM描述残差 | supported candidate（当前无专家H=77.808093） | `INNOVATION-024_clre/` |
| V2-INNOVATION-025 | IDEA-059 | CLEC跨LLM全局与局部组合 | rejected（低于CLRE） | `INNOVATION-025_clec/` |

`INNOVATION-MODULE-1 / TG-VPR-H1`由owner直接提升为`FRAMEWORK-V2`，不占用本编号。旧H1证据只作为`experiments/v2/evidence/legacy_h1/`下的`legacy_ref`。
