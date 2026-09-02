# IDEA-218 / Class-Conditional MIL Visual Expert (CCMVE)

- status: `revised_after_prequeue_gate`
- problem_category: `visual_representation`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-217 / PMVE`

CCMVE让每个whole-class mean8 prototype分别对36patch做log-mean-exp MIL，训练rank16共享残差投影并输出完整视觉类分布。它只作为operational V，不声称与S统计独立。

- old_solution_path: class-independent pooling丢失候选特定证据。
- new_solution_path: class-conditioned patch MIL -> full local-visual distribution。
- non_equivalence_test: 必须胜过class-name、CLS-only、frozen、max-patch和mean-patch控制并保持unseen prototype geometry。
- minimal_viability: V H/ZS、独有纠正、oracle空间、融合与bootstrap门。
- current_advantage: 有真实错误互补但base整体Gate未通过；`performance_status=proof_of_path`。
- failure_boundary: seen CE可扭曲unseen文本几何；LME不一定胜过简单max。

## 审查与结果

- Idea独立A/B：`d447781ac042b49789f7f6e31e350d509831d9462c59b6f3947edb10997fb8e0` / `7ec388d0784a1a5b7e69d4e71d17c14ddf3e0e63ee7178e6d4334e200ab1fdfd`；交叉A/B：`e5cf023d67f65738f0d5d433a2eea213ea580a980f0ca11d29efa5db75c6477d` / `943c1693cbede228f32179343ff65a9e3502a6182b96ef98ee60042dd43cc33b`。
- Gate脚本SHA：`159d28b98f650ee19cbedbb3032fdffc06bdae635e9e69622c53978995ef59c2`；代码复核P0/P1=0。
- 结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/ccmve/IDEA-218-GATE0/result.json@sha256:07c40b3e43410fc727110d31633c39d7ac8b1f940542235617c3ee4e11afb864`。

CCMVE V为`U/S/H/ZS=33.563895/47.782080/39.430414/54.778696`。seen/unseen中S错V对=`139/144`张，覆盖`58/26`类；oracle `H=75.077573`，比mean8 S高`+6.327007`且bootstrap下界`+4.699`，首次证明36patch视觉分布具有广泛互补错误。

但base Gate失败：equal-logit融合仅`H=56.341914`；max-patch控制`H=42.258276`高于CCMVE；投影后unseen prototype与原型cosine mean/min=`0.4893/0.1464`，严重破坏语义几何。下一补救冻结文本原型，仅训练patch投影，并采用top-patch删除一致性/第二证据。
