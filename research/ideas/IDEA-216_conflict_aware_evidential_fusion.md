# IDEA-216 / Conflict-Aware Evidential Fusion (CAEF)

- status: `rejected_at_prequeue_gate`
- problem_category: `visual_semantic_interaction`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- source: IDEA-213/214/215 role-patch grounding failures

## Hypothesis

CAEF将交互改为多视图证据仲裁：S是mean8语义分布；V用CLS引导、完全不看文本的36patch attention-MIL产生视觉分布；I按Subjective Logic显式建模belief、ignorance与conflict，并用Dempster规则融合。它借鉴Trusted Multi-View Classification（ICLR2021）、Provable Dynamic Fusion（ICML2023）、ICCV2023 confusion/ignorance和CVPR2024 selective VQA。

- old_solution_path: 强迫role-patch定位后再加margin。
- new_solution_path: semantic/visual opinions -> conflict/ignorance state -> non-additive fusion。
- principle_difference: I仲裁两个预测分布的冲突，而非生成局部相似度。
- non_equivalence_test: 必须胜过equal-logit/product、probability-average、entropy-only和单视图。
- minimal_viability: V存在S没有的纠错样本、oracle H空间>=3、entropy reliability AUC>=0.55，Full与三个off全部净增。
- current_advantage: none；`performance_status=proof_of_path`。
- failure_boundary: 两个view共享CLIP空间且V可能没有独立分类信息。

## 双Agent与Gate

- Idea独立A/B：`db6a17777204575d3910d52a684e351ce9da6ca0aea07a1ced89b4de0692eb42` / `56a1bdbd5c41cacda7b029b63005b6ed8c6bbc56216cda2127dd04f9ab29385d`。
- Idea交叉A/B：`6deafb1eb9c89e6290ec3bd6ef0a2e31c32f86279689fd78022b2a4a84980035` / `cde88bd12326b6179a20f02680d2c60bfc153e86e0eec62e0f7781a198dfb675`。
- Gate脚本SHA：`b34940991f175871525d82a43a0511d307dc18cc53b3ea6dbc963ed64dfa367d`；修复后代码复核P0/P1=0。
- 结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/caef/IDEA-216-GATE0/result.json@sha256:eec9d4e86fb5e6dc3f0b4974a8d62531e8704d088505b24236274daaa135d7a9`。

## Result and decision

- S：`H=68.750566`；V：`H=1.276588`、`ZS=12.807550`。
- unseen中`S_wrong/V_correct=0`，seen仅14；oracle `H=69.093950`，相对S只有`+0.343384`，不足以支撑现实`+1`融合。
- equal-logit/product `H=53.264720`，probability-average `H=66.860049`，Dempster Full `H=57.441777`；entropy-only退化为S，`H=68.750566`。
- Dempster Full相对V-off(S-only)为`-11.308789 H`；绝大多数硬门失败，`gate_pass=false`。

CAEF base被否定。失败不是融合公式选择问题，而是V没有独立信息。补救必须先训练text-free视觉池化器并证明S/V错误互补；在此之前禁止继续调Dempster温度、ignorance或冲突权重。
