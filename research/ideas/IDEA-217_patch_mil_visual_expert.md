# IDEA-217 / Patch-MIL Visual Expert (PMVE)

- status: `rejected_at_prequeue_gate`
- problem_category: `visual_representation`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-216 / CAEF`

## Hypothesis

PMVE用seen CE训练完全不看文本的36patch attention-MIL，再将池化视觉向量与mean8 prototypes分类，目标是建立能纠正S错误的第二视觉view。方法降级为operational patch-bag view，不声称foreground或统计独立。

- old_solution_path: frozen text-free V没有独立分类信息。
- new_solution_path: seen label -> learned patch-bag attention -> visual opinion。
- non_equivalence_test: 必须胜过frozen、uniform、no-CLS、CLS-only、shifted-training与uniformized控制。
- minimal_viability: V H>=30/ZS>=40，seen/unseen各至少50个S错V对、oracle空间>=3、融合非破坏和bootstrap通过。
- current_advantage: none；`performance_status=proof_of_path`。
- failure_boundary: class-independent单向量池化可能丢失类别特定局部证据，attention可塌缩到无判别力patch。

## 双Agent与结果

- Idea独立A/B：`120ef45f91e3eae7f5f16dfbc22a9b5e9fb04a54ff41ade6250d5540ecf94b78` / `ff4975f3edefb7fb5a6e70fd1fce4795bd9132d3e58bc4e55bb517099e2ca27f`。
- Idea交叉A/B：`7a778ca000c2c5282584f068bf3303438191a95089ccd7a8ac1ee6b9e244d221` / `b7ddda292af94b3fb234116f403a5304ed683b7a6bd0dbc561ade3e90e0ac164`。
- Gate脚本SHA：`49bcd8c61526a53aa88092e0b3f9de793017b09b78aae24d655faae790ef52a9`；代码复核P0/P1=0。
- 结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/pmve/IDEA-217-GATE0/result.json@sha256:bb484b2711626b53717cc3511403cae4e5b1eaca321b2adab3d3cb35f50a7391`。

## Evidence and decision

- PMVE V：`U/S/H/ZS=2.905085/5.681084/3.844329/22.851364`；mean8 S为`H=68.750566`。
- S错/V对：seen30张/9类，unseen10张/3类；oracle `H=69.682473`，相对S仅`+0.931907`，bootstrap下界`+0.3545`，不足以支撑现实+1融合。
- equal-logit `H=56.022888`，相对S严重损坏。
- learned attention熵约0.18、有效patch数约2.2，但PMVE与no-CLS结果几乎相同；CLS条件路径无作用，尖锐attention不等于有效视觉证据。
- 全部性能/互补硬门失败，仅非均匀attention与micro梯度通过，`gate_pass=false`。

PMVE被否定。下一救援不能继续把36patch压成一个类无关向量，而应为每个类别分别做MIL证据聚合，形成class-conditional visual distribution后再测试CAEF。
