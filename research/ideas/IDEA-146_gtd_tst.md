# IDEA-146：测地目标蒸馏切空间迁移（GTD-TST）

- 状态：`supported_promoted_to_framework_v4`
- origin_framework：`FRAMEWORK-V3-EXPLORATION`
- tested_in：`V3-TRY-022 / V3-TRY-040 / V3-TRY-041 / V3-TRY-046 / V3-TRY-047`
- promoted_to：`FRAMEWORK-V4`
- problem_category：`cross_class_transfer`
- mechanism_tags：`[geodesic_transport, target_distillation, unseen_prototype]`
- source_type：`code_analysis + experiment_result + first_principles + owner_hypothesis`
- 父条件：`V3-TRY-002 / TG-only / U/S/H/ZS=78.407878/74.983871/76.657659/86.146760`
- 准确base commit：`cd30797a5eab3aa6ed28bd04df0b17f413730063`
- 问题：旧signed/all-class TST没有稳定独立1H增益；冻结的正步长Faithful-TST又无法表示严格零移动，并以随机batch的pseudo-unseen CE间接训练Gate，不能明确区分“Value方向有害”和“Gate没有学会”。
- 核心假设：seen视觉中心可以在`Mean8→Value`精确测地线上产生可审计的`CE+0.1θ²`最优角度；将该角度蒸馏给只读六维文本几何Gate，并只迁移true-unseen原型，能够在不使用unseen图像梯度的情况下相对TG提高至少1H。
- 唯一核心改动：`Mean8/Value文本几何 → 33点球面oracle → SmoothL1共享Gate → true-unseen-only测地迁移`。seen原型保持TG；Gate零初始化和dead-zone保证`θ=0`逐元素复现父TG。角度上限固定为`min(Mean8到Value的球面弧长, atan(1.5))`；`atan(1.5)≈56.31°`是把历史最大切向step 1.5转换成球面角上限，不是重新搜索出的超参数。
- 训练：CUB、seed7、batch50、fixed150（精确21,171 updates）；从V3-TRY-002 checkpoint开始，TG与Gate从update1一段式联合更新。只用7057张trainval在update `1+141k,k=0..149`刷新150次seen teacher；每次绑定model/package SHA、fold ID、target/mask/gain。每141步、21,150和最终21,171评估，共152点，不早停；checkpoint保存teacher与CPU/CUDA/batch RNG并支持同RUN显式resume。
- 筛选与成立采用固定三态：`ΔH<0.8`或`|U-S|>=8`直接drop，此时不绑定匹配条件；`0.8<=ΔH<1`且gap合格时输出`trigger_try020_static_below1`，触发TRY-020但`static_support_passed=false`，不得晋级；`ΔH>=1`且gap合格时也只输出`pending_matched_try020_comparison`并置`static_support_passed=true`。后二者均要求TRY-020，只有`H(TRY-022)-H(TRY-020)>=1.000`且共同满足gap门槛，才能说明GTD具有独立增益并进入fixed200与完整模型`-GTD`验证。TRY-020当前只是条件触发项；真实执行前必须另行冻结其准确config、RUN commit和queue行，不能看到TRY-022结果后再定义训练语义。
- 失败：仅当固定150轮`ΔH<0.8`或U/S差达到8时直接drop；`0.8<=ΔH<1`不是支持成立，但必须进入预注册TRY-020匹配控制。oracle目标大面积为0、Gate拟合失败和seen→unseen迁移失败只用于区分失败原因，不在同一Experiment临时改网格、惩罚或dead-zone。
- 结果：TRY-022完整150轮best位于update846/eval6，`U/S/H/ZS=80.559021/75.587094/77.993901/86.611593`，相对静态TG `ΔU/ΔS/ΔH/ΔZS=+2.151144/+0.603223/+1.336242/+0.464833`，U/S gap=`4.971927`。已过静态1H与gap门槛，按预注册等待TRY-020匹配控制，未宣称最终支持。
- 风险诊断：best时true-unseen move rate=`1.0`、平均theta=`15.84°`，seen Gate target correlation=`-0.228`；虽H显著提升，但Gate解释性和全移动倾向存在风险，必须结合TRY-020及后续`-GTD`消融判断。
- 三创新接口：TG负责seen监督下的角色原型建立；GTD-TST只学习从seen视觉几何推断unseen原型该沿语义切向移动多少，形成“表示建立→迁移决策”的连续链。
- 代码语义commit：`fee829bd56b6ac9b366da82e1438b9d7bee872a8`
- evidence_refs：V3-TRY-002 TG父指标；V2 signed/all-class TST不足1H的本仓库结果；IDEA-145 Faithful-TST代码边界观察；owner确认的GTD-TST假设。

## 严格从头训练诊断

本地V3-TRY-040/041使用同一seed7随机TG初始化、同一批次序列和固定150轮，不加载历史TG checkpoint。TG-only与TG+GTD均在epoch80达到global best；GTD使H从`76.649020`提高到`78.119641`，匹配RUN及同checkpoint移除增益均为`+1.470620 H`。两条件的TG父轨迹在全部152个评估点逐项一致，说明增益来自GTD而非父模型随机漂移。该诊断尚待项目两轮独立审核后才能晋级正式证据。
