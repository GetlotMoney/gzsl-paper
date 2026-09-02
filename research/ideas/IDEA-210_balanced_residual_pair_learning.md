# IDEA-210 / Balanced Residual Pair Learning (BRPL)

- status: `contingent_rescue_proof_of_path`
- problem_category: `learning_generalization`
- mechanism_tags: `semantic_dominance`, `balanced_pair_loss`, `simultaneous_residual_training`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-208 / V6-TRY-010`, rescue `1/3`

## 问题与可证伪假设

CTPM 的正式最佳点把 H 从 class-name Parent 的 `63.192339` 提至 `69.320040`，但相同 checkpoint 的 S/V/I 关停差分别为 `55.951756/0.233980/0.394349`。全局语义残差吸收了几乎所有可学习误差，视觉与交互虽参与训练却没有形成独立决策贡献。

BRPL 保持 CTPM 的线上模型完全不变，以同一 Top1/Top2、8×36 注意力和一次连续反对称修正推理。训练时语义采用较慢学习率；视觉只在冻结语义前缀上学习平衡 c1/c2 候选对损失；交互在冻结 S/V 前缀上重新计算并只接收自己的平衡候选对梯度。若同一 Full 最优 checkpoint 的三个 H gap 都不能达到 `+1.0`，该救援失败。

## 路径与边界

- old_solution_path: 一个共享学习率、Full CE 加非平衡 pair CE，让所有分支竞争同一候选对误差。
- new_solution_path: 一个同步优化步骤，Full CE 保持全局约束；V 与 I 各自从 stop-gradient 前缀接收 equal-mass c1/c2 残差监督。
- principle_difference: 标签误差被分配给仍未解决的后续分支，而非由最容易的语义分支统一吸收。
- non_equivalence_test: 成功后，BRPL 的最小 S/V/I gap 必须比同构的非平衡分支损失和共享学习率控制各高至少 `0.5`；否则不作方法 claim。
- minimal_viability: 冻结 seed-7 采样轨迹首个同时包含 c1/c2 的 batch，验证 Full CE 到达 S/V/I，V auxiliary 只到 V，I auxiliary 只到 I，且反对称 scatter 与 off 路径不变。
- minimal_falsification: 一次固定 `batch=50`、`28,228` updates 的 Chen-style official-test-selected 运行，记录辅助损失跳过率并要求 V/I 均不超过 `5%`。
- current_advantage: 尚无；这是 CTPM 失败驱动的救援，`performance_status=proof_of_path`。
- failure_boundary: c2 稀少或含噪时，平衡损失可能过度放大；冻结前缀仍可能使 V/I 学到重叠修正；Top2 外真类仍主要依赖 role logits。

不宣称新的推理范式或论文创新点。该救援只检验“分支隔离且平衡的残差监督”是否能把既有 CTPM 的三模块合同变成真实结果。没有教师、蒸馏、专家属性、未见图像梯度、PCLR 在线推理或 HTML 图。

## 双 Agent 对抗定稿（2026-09-02）

- 最终草稿 SHA256: `8f948dad1697689400f1d971a2c4971c598e63533fd1eac998aecac77186d91d`
- Round2 独立 A/B: `7ce808158fec3e39882e389f7d977b6e13728d40ca552bed38eb5e93bc824583` / `a90ed5031cfa2f6ee8fd633ead6428839d4a71a815ad13e5bccfbc3c6fcdbb09`
- 最终交叉 A/B: `890b33794e49fa0033b6141bfccf13995f4c8d89ecb227df214b52f0e37328a6` / `f779339297bb377eabf0e10e48c5484101c5b843b3aa1a227c93be0344d29c0a`
- 共同结论: `P0=0 / P1=0 / P2=0`，**范式Idea双Agent对抗审核通过**；仅授权本次 proof-of-path 救援实现。

CTPM 失败收据：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/ctpm/V6-TRY-010/metrics.json@sha256:0150246204204524ad6224ff36a261c65800f6221b433b3a9c31c523aad50234`。
