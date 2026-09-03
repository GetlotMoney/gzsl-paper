# IDEA-232 / Class-level Residual Regression (CRR) — 专家属性主线终结性 kill-gate

- status: `proposed`
- idea_id: IDEA-232
- problem_category: `expert_attribute_candidate_verification`
- mechanism_tags: [class_level_regression, closed_form_ridge, prototype_residual, expert_attributes]
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: 无（不是IDEA-231的补救；是IDEA-226/227/229/230/231之后的属性主线终结检验）
- performance_status: `not_yet_tested`

## 定位（双Agent对抗收敛结论，2026-09-03）

**诊断/kill-gate，不是创新候选。** 对抗审查确认：`p_c = Norm(t_c + β·q_c W)` 相对父框架 mean8 是学习型加性校准残差（AGENTS.md 默认非范式清单命中），且可还原为 ESZSL（ICML 2015 类级闭式ridge）+ TIP2017 原型矫正 + Candle 式原型融合的组合。不进三创新主线，不写 paper_level_claim。

close_paradigm_work（六篇，逐篇区别）：
- ESZSL（Romera-Paredes & Torr, ICML 2015）：类级闭式 ridge 语义→视觉回归同构；CRR 差异仅残差目标+文本锚定，属工程约束组合非范式差异。
- Luo et al. TIP 2017（属性回归+类原型矫正，CUB）：概念重合。
- Distribution Calibration（ICLR 2021）：seen 视觉统计校准新类原型同思路。
- Candle（KDD 2024）：CLIP 冻结特征 text/visual 原型混合。
- Prototype Rectification（ECCV 2020）：基类统计矫正新类原型。
- AttributeSelect（ECCV 2026 workshop）：裸属性 prompt 59.5%→15.5%，印证弱属性证据不能独立支撑。

## 问题与假设

- problem: 312 维类级专家属性至今未产生合法可迁移增益（五连败）；核心障碍是"符号属性→CLIP 图像空间类别方向"的转换未被正确解决。前五案都把属性逼去解图像级强判别任务，违反其类级信噪比量级。
- hypothesis: 属性信噪比与"类视觉中心相对文本原型的残差"（类级、图像噪声已被平均消除）量级匹配；存在稳定线性映射 W 使 q_c 回归 r_c 的可泛化成分，且该成分能安全注入文本原型产生 OOF 增益。
- 关键证据（两名Agent独立只读诊断交叉确认，不作为evidence_refs，正式Gate重跑）：OOF Δ +1.16~+2.18pp（4种子）、置换对照强分离（R²≈−0.9 / 混合Δ≈−3.2pp）、纠错集中在语义近邻对（纠正对cos=0.793 vs 类间均值0.540，背景混杂反例被削弱）。

## 冻结预注册合同（本卡即合同，任何字段不得在结果出来后修改）

### 数据资产（SHA在Gate运行时回填）

- CLIP 特征与8句嵌入：`/data/lby/projects/cv_project/GZSL_Warehouse/assets/clip_vitl14_336/CUB/69c9c6d82a755fe8/`（train_features/train_labels、role_sentence_embeds、class_names.json）
- 类级专家属性：`/data/lby/projects/cv_project/GZSL_Warehouse/datasets/splits/xlsa17/data/CUB/att_splits.mat` 的 `att`（312×200），按类名与 class_names.json 对齐
- 类对齐是关键正确性风险：脚本必须打印对齐诊断（匹配类数、未匹配类），审查重点

### 公式（冻结）

```text
t_c  = L2(mean_8句(role_sentence_embeds[c]))        # 句子直接均值→行L2，禁止句子预归一化
μ_c  = L2(mean(train_features[labels==c]))           # 先均值后归一化，raw未归一化特征
r_c  = (I − t_c t_cᵀ) μ_c                           # 去t_c平行分量
折内: r_c ← r_c − mean_{c∈train_fold}(r_c)          # 折内中心化
y_c  = r_c / ‖r_c‖                                   # 单位方向目标
x_c  = q_c − mean_{c∈train_fold}(q_c)               # 原值仅减均值，禁止z-score
W    = (XᵀX + λI)⁻¹ XᵀY                             # 全维312，无谱截断
p_c  = L2(t_c + β · L2(x_c W))                       # 推理（折外类x_c减同一折内均值）
```

### 实验设计（唯一变量：折外类原型是否加预测残差）

- 150 个 trainval seen 类，种子 [7, 11, 33, 55]，每种子 `randperm(150)` 分3折（每折外50类）
- 每折：折内100类训 W；**150类竞争场中折内100类原型保持 t_c 不修正，仅折外50类**原型变为 `p_c = L2(t_c + β·L2(x_c W))`
- 评估：折外50类的 trainval 图像（该类全部图），150类竞争，CBA=折外类 macro accuracy
- 每种子3折覆盖150类；4种子合并。基线（in-run）：完全相同管线但折外类原型 = t_c
- 主口径 trainval（不碰 test）；secondary 披露 test_seen 1764张同构造口径（不进判据）

### 超参（预注册固定点+网格披露）

- λ* = 0.1，β* = 0.1（第一轮交叉前冻结，刻意非峰点，λ=0.3为双方独立网格峰但全平台0.03~1.0过+1pp，不换点）
- λ 网格 {1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10} × β 网格 {0.05, 0.1, 0.15, 0.2, 0.3}，全网格仅披露稳定性，判决只用固定点

### 主判据（预注册点 (λ*=0.1, β*=0.1)）

1. 4种子合并均值 Δ ≥ in-run 基线 + 1.0pp
2. 每种子 ≥2/3 折为正
3. class-level paired bootstrap（150类重采样，10000次）95% CI 下界 > 0，仅在预注册点计算
4. 最差种子强制披露：<+0.5pp 标 instability warning（非阻断）

### 预注册对照（与主条件同管线同折同网格）

- 乱序对照：q_c 类间置换（≥5个置换种子），预期 Δ 掉至≈0或负
- 类频率对照：用 n_c（类内样本数向量）代替 q_c
- 维度匹配文本对照：t_c PCA→312维后代 q_c
- oracle 上限：折外类原型加真实 r_c（折外类 μ_c 仅用于上限，不进管线）
- μ_c 上限：折外类原型直接 = μ_c
- in-sample/折外分解披露：另跑全拟合（150类训W、150类全修正）在 trainval 与 test_seen 上，量化记忆成分（已知约3/4增益来自in-sample，+6.8pp量级 vs 折外+1.5pp量级）

### 两级失败边界（无续命空间）

- Level 1：OOF Gate 任一主判据不过 → **专家属性主线永久放弃**（不调 λ/β/网格/预处理/训练方式，不再有属性补救案）
- Level 2：OOF 过线 → 进入 chen_shiming_code_aligned_test_selected_gzsl 正式协议（200类 U/S/H/ZS，整模型H选择）后，若相对 mean8 同口径 **ΔH ≤ 0 或 ΔU < 0 → 专家属性主线同样放弃**；失效时分叉诊断：U降S涨=不对称/校准失效 vs U降S不涨=外推崩塌/混杂
- β 在 official 阶段沿用不重选；任何 test 侧重选标记 `nested_official_test_selection: true`

### 已知风险披露

- S/U 修正强度天然不对称（in-sample +6.8 vs 折外 +1.5），单一β无法双侧最优（OOF峰0.1，in-sample 单调到≥0.2）
- unseen 50类属性向量在 seen 属性支撑外存在外推（实测范数比值0.956，尾部个别类max 1.127）
- 残差含属性解释不了的成分：类频率(spearman −0.218)、成像条件、类内多态性（雌雄异色类μ_c语义模糊）
- 纠错集中于语义近邻对支持细粒度语义归因，但"属性=物种指纹与拍摄条件相关"的残余解释不能彻底排除（kill-gate判决不依赖语义纯度；语义归因是论文阶段前置条件）

## 审查记录

- 方案级：2026-09-03 两名临时子Agent完成双Agent对抗（独立审查→交叉质询→第二轮聚焦），双方最终均判 pass，无剩余实质分歧；因已降级为诊断，不适用"范式Idea双Agent对抗审核通过"定稿短语。
- 代码级：单Agent自问自答两轮（owner简化指令），待pre-run commit后执行，结论回填于此。

## 结果（2026-09-03 post-run回填）

### 运行身份

- 正式运行commit：`101f967`（seen集合修复后）；预注册冻结commit：`cc0e6ec`；分支 `exp/v5/diagnostic/idea-232-crr`
- 结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v5/crr/IDEA-232-GATE0/result.json@sha256:2ee5c54be33143aab153dc825fb791878dc054c0e8a3047f9f930c7b299e0f61`
- 对称口径检查：`.../symmetric_check.json@sha256:5e9c1541df3f09be1cf06e2087532a7d63bf3f6766a8b8d5f165603b478fa753`
- 审查：单Agent自问自答两轮通过（P0=0/P1=0，8项P2披露性修复）；运行中发现并修复一处P0级数据身份错误（见下）

### 关键数据身份发现（运行时断言拦截）

本 split 的 seen/unseen 划分**不是**"标签0-149=seen"：`001.Black_footed_Albatross`是seen（标签150）、`012.Yellow_headed_Blackbird`是unseen（标签6）。seen集合由 `unique(train_labels)` 推导，经 `res101.mat` image_files 路径逐类验证（enc==allclasses_names顺序，200/200命中）。资产标签空间=res101编码−1=att列序=8句嵌入行序，att对齐无需置换（perm=identity）。

### 原口径结果与口径缺陷（主判据无效）

原口径（折外50类修正 vs 折内100类保持t_c）：base 71.13% → 83.69%，Δ=+12.56pp，四种子+10.7~+13.7pp，乱序−2.8~−4.4pp，CI [10.3,14.9]pp——表面全过。

**但该口径存在系统性竞争场不对称偏置**：折外类原型被残差拉入图像锥后与图像余弦整体升高，折内类保持文本锥t_c分数不升；评估图像全部属于折外类，即使残差方向信息完全错误，只要原型被拉近图像锥CBA就涨。证据：text_pca对照（用文本PCA回归残差，无属性信息）在同一不对称口径下+11.32pp，几乎等于属性版+12.56pp——不对称口径增益与回归器质量几乎无关。**原口径主判据（含+12.56pp、45/45网格、上限锚点）全部作废，不得引用。**

### 对称口径结果（修正判据，待owner裁决是否采纳为正式口径）

对称口径：每类用其作为折外类时该折训练的W修正，竞争场150类全修正 vs 150类全t_c（无不对称偏置）：

| 条件 | trainval Δ | test_seen Δ | bootstrap 95% CI |
|---|---:|---:|---|
| **attr（主条件 λ*=0.1 β*=0.1）** | **+1.086pp** | +1.208pp | [0.152, 2.080] |
| text_pca对照 | −1.515pp | −1.274pp | [−2.421, −0.639] |
| 类频率对照 | −5.918pp | −5.831pp | [−8.799, −3.248] |
| 乱序attr（3个置换种子） | −4.1~−5.0pp | — | — |

β扫描（对称口径，λ=0.1）：0.05:+0.76 / 0.1:+1.09 / 0.15:+0.91 / 0.2:+0.43 / 0.3:−1.41——预注册点0.1恰为峰，平台0.05~0.2全正但仅0.1过+1pp。

in-sample全修正披露：trainval +5.84pp / test_seen +5.33pp；对称OOF +1.09pp——兑现率约19%（与对抗诊断估的20%一致，约3/4为类级记忆成分）。

对照结构干净：属性版唯一为正且过线，文本/频率/乱序全部显著为负——信号特定于"属性→残差"配对，非文本冗余、非频率代理、非配对噪声。

### 状态与待裁决

- c1（均值≥+1pp）：+1.086pp 贴线过；c3（CI下界>0）：0.15pp 过；c2/c4（折级/最差种子）对称口径下未正式计算（对称检查脚本为初步版）。
- **待owner裁决**：(a) 采纳对称口径为正式口径，重跑对称版完整Gate（c1-c4全判据）作为正式判决；(b) 判原合同口径无效即gate fail，属性主线放弃；(c) 其他裁决。
- 口径修正是评估语义变化，按项目规则不经owner确认不得自行替换主判据。

### Owner 裁决（2026-09-03，方案a）

owner 采纳对称口径为正式口径，重跑对称版完整 Gate 出正式判决。对称判据定义（c1-c4 语义适配）：

- **对称口径**：每类用其作为折外类时该折训练的 W 修正原型，竞争场 = 150 类全修正 vs 150 类全 t_c 基线，评估 = 全部 trainval 图像 / 150 类 macro。基线（in-run）与种子无关。
- c1：4种子整体 delta 均值 ≥ in-run 基线 +1.0pp
- c2：每种子 3 折级中 ≥2 折为正（折级 = 该折 50 类在全修正场 vs 全基线场的 macro 差；竞争场每种子统一，折间差异来自评估类子集）
- c3：class-level paired bootstrap（150类 × 跨4种子均值差，10000次）95% CI 下界 > 0，仅预注册点 (λ*=0.1, β*=0.1)
- c4：最差种子披露，< +0.5pp 标 instability warning（非阻断）
- 对照同前（乱序×5、类频率、文本PCA、oracle r_c 上限、μ_c 上限、in-sample 披露），全部改对称口径；网格披露 seed=7 单种子（同原合同）。文本PCA对照因 symmetric builder 接口限制改为全150类PCA拟合（~100维），披露口径。
- 正式脚本：`tools/idea232_crr_gate_symmetric.py`，复用主脚本已审查的 `load_assets / build_fold_targets / ridge_fit / eval_cba / run_insample`。

### 对称版正式判决（post-run回填）

待运行。

