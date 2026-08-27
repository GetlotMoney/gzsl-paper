# IDEA-146：测地目标蒸馏切空间迁移（GTD-TST）

- 状态：`testing`
- source_type：`code_analysis + experiment_result + first_principles + owner_hypothesis`
- 父条件：`V3-TRY-002 / TG-only / U/S/H/ZS=78.407878/74.983871/76.657659/86.146760`
- 准确base commit：`cd30797a5eab3aa6ed28bd04df0b17f413730063`
- 问题：旧signed/all-class TST没有稳定独立1H增益；冻结的正步长Faithful-TST又无法表示严格零移动，并以随机batch的pseudo-unseen CE间接训练Gate，不能明确区分“Value方向有害”和“Gate没有学会”。
- 核心假设：seen视觉中心可以在`Mean8→Value`精确测地线上产生可审计的`CE+0.1θ²`最优角度；将该角度蒸馏给只读六维文本几何Gate，并只迁移true-unseen原型，能够在不使用unseen图像梯度的情况下相对TG提高至少1H。
- 唯一核心改动：`Mean8/Value文本几何 → 33点球面oracle → SmoothL1共享Gate → true-unseen-only测地迁移`。seen原型保持TG；Gate零初始化和dead-zone保证`θ=0`逐元素复现父TG。
- 训练：CUB、seed7、batch50、精确21,171 updates；从V3-TRY-002 checkpoint开始，TG与Gate从update1一段式联合更新。每个名义epoch只用7057张trainval刷新seen teacher；每141步、21,150和最终21,171评估，共152点，不早停。
- 成立：TRY-022相对固定父TG `ΔH>=1.000`且`|U-S|<8`，并且真实unseen没有图像梯度；通过后再跑匹配fixed200 TG-only与`-GTD`单模块移除。
- 失败：完整150轮未达到1H，或U/S差达到8，直接标记drop；oracle目标大面积为0、Gate拟合失败和seen→unseen迁移失败仅用于区分失败原因，不在同一Experiment临时改网格、惩罚或dead-zone。
- 三创新接口：TG负责seen监督下的角色原型建立；GTD-TST只学习从seen视觉几何推断unseen原型该沿语义切向移动多少，形成“表示建立→迁移决策”的连续链。
- 代码语义commit：`550a36a9483a2c8dec67fc8a8240f259d215b83b`
- evidence_refs：V3-TRY-002 TG父指标；V2 signed/all-class TST不足1H的本仓库结果；IDEA-145 Faithful-TST代码边界观察；owner确认的GTD-TST假设。
