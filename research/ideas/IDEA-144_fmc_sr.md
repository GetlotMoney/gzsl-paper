# IDEA-144：频域引导的多尺度候选安全纠错

- 状态：`testing_rescue_2_pending`
- 父条件：`V3-TRY-002 / TG-only / U/S/H/ZS=78.407878/74.983871/76.657659/86.146760`
- 问题：IDEA-142表明频域token对部分unseen有信号但无条件使用会破坏seen；IDEA-143表明冻结文本差方向即使经过完整576 patch、多尺度和硬拒绝仍整体有害。失败共同指向：视觉证据必须同时学习“稳定区域、候选差异和最小修改”。
- 核心假设：频域vote作为类别无关的稳定区域先验，与TG Top-3候选的六角色差异证据在24/12/6尺度中相交；共享反对称pair网络学习跨类别修正，最小干预训练只在TG错误且真类位于Top-3时执行刚好翻转，能在保护正确TG样本的同时获得至少1 H提升。
- 计算链：`TG Top-3 → frequency vote prior × candidate-difference evidence → 18维多尺度pair特征 → shared antisymmetric MLP → learned safety gate → zero-sum Top-3 correction`。
- 结构边界：完整576 projected patch；repo/K=1频域vote；前六局部角色；24/12/6；pair MLP=`22→16→1`；gate=`4→8→1`；最大pair修正0.25；输出层零初始化，初始严格复现TG；非候选类别不变。
- TRY-014：相同结构使用ordinary CE，检验视觉结构本身。
- TRY-015：相同结构使用minimal-intervention correction，检验训练框架；`L=TG CE+topology+flip+0.2 keep+0.2 abstain+0.1 consistency+0.01 residual`。
- 首次结果：TRY-014/015均在旧预注册第50点评估规则下停止，global best严格保留update0父TG；最佳已训练H分别为`76.633457/76.595645`，均低于父模型。视觉残差在最佳已训练点未改变Top-3预测，指标变化主要来自TG继续训练。
- 固定150补救：V3-TRY-016已由并行分支Faithful-TST-E2E占用，017为其潜在补救预留；FMC-SR改为TRY-018/019。从TG父checkpoint开始，TG与视觉参数从第一个更新同时可训练，CLIP/patch/vote冻结；batch50、seed7、精确`21,171`次更新。前5名义epoch TG保持`1e-5`、视觉从`1e-5`warmup到`1e-4`；随后TG `1e-5→1e-6`、视觉`1e-4→1e-5`分别余弦退火。update 0、每个141步点至21,150、最终21,171均评估，共152点，不提前停止。
- 成立：TRY-018相对TG `ΔH>=1.000`证明结构；TRY-019相对匹配TRY-018 `ΔH>=1.000`证明训练框架；共同要求U/S差小于8。胜出条件再固定200轮和单模块移除。
- 失败：两个条件均未过门槛则该复合视觉轴关闭；不得继续调beta、hidden或损失权重。
- 固定150结果：TRY-018 global best严格为update0父TG，`U/S/H/ZS=78.407878/74.983871/76.657659/86.146760`；最佳已训练H=`76.633457`。TRY-019 best在update564，`U/S/H/ZS=77.173418/76.225722/76.696642/86.146760`，相对TG和匹配TRY-018均仅`+0.038983 H`，低于最低0.8门槛。
- 诊断：TRY-019 best的视觉残差相对同checkpoint联合TG没有翻转任何预测；完整模型相对冻结TG为seen净+16、unseen净-37，说明微小H变化主要来自TG继续训练，视觉分支没有形成有效决策贡献。FMC-SR第1次完整150轮补救失败，按owner规则仍可进行最多2次有明确原因的补救；不得把本结果包装为有效模块。
- 代码语义commit：`9f38d4d87e2ec98e047d8cad0364cf0dd61cf5f6`
- 分支基线：`exp/v3/innovation/innovation-005-fmc-sr`从准确TG父commit `cd30797a5eab3aa6ed28bd04df0b17f413730063`创建；实现前`model/tools/tests`相对父commit无差异，只继承公共规范和历史账本。
- evidence_refs：IDEA-142/TRY-012；IDEA-143/TRY-013；owner确认的统一交互与最小干预训练假设。
