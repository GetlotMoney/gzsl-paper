# FRAMEWORK-V2 长期创新实验计划

## 目标与事实起点

```yaml
base_framework: FRAMEWORK-V2 / TG-VPR-H1
supported_innovations: [TG-VPR, TST, CCGR]
current_best_observation: V2-TRY-078 / TG-VPR + TST + NTR + CCGR
current_best_H: 77.572682
current_best_seed: gate_training_seed_17_on_data_seed_7
completed_try_count: 83
minimum_total_try_count: 50
target_best_H: 78.0
evaluation_protocol: test_selected_inductive_gzsl
test_used_for_selection: true
unseen_images_used_for_gradient: false
```

主成绩使用最高seed H；mean/min/max/range只用于稳定性诊断。`H range<=1.0`个百分点视为差距不大，可采用最高seed。所有新核心创新必须提供module-off消融、准确父条件、训练边界和HTML框架图。

## 第一性原理框架

当前单点原型和普通联合训练存在四个根问题：类别内部不确定性未建模、seen到unseen只有单一路径、pseudo-unseen没有真正的外层优化、多个训练目标会产生梯度冲突。后续按下列连续逻辑搜索：

```text
八角色视觉描述
  -> TG-VPR结构化语义
  -> 概率/多原型表示
  -> 图结构或几何迁移
  -> 双层pseudo-unseen元训练
  -> 不确定性感知联合分类
```

## TRY-037至TRY-064

### BMR：双层元路由，TRY-037至TRY-040

- `TRY-037`：内层只用100类pseudo-seen训练临时原型，外层50类pseudo-unseen CE只更新迁移Gate；一阶双层梯度。
- `TRY-038`：若外层无提升，加入内外层梯度余弦诊断与冲突投影。
- `TRY-039`：若难折不稳定，将固定mod-3折改为语义距离控制的hard episode。
- `TRY-040`：若跨seed不稳定，训练3个外层Gate并平均；仍失败则停止BMR。

### DPT：概率分布式原型，TRY-041至TRY-044

- `TRY-041`：每类由单位均值方向和vMF浓度表示；浓度由八角色文本离散度确定。
- `TRY-042`：若固定浓度不足，使用类别几何Gate学习有界浓度。
- `TRY-043`：若单峰不足，使用local与global两个分量的混合分布。
- `TRY-044`：若分量塌缩，加入分量间最小角距离；仍失败则停止DPT。

### SGT：语义图迁移，TRY-045至TRY-048

- `TRY-045`：在200类文本KNN图上传播seen训练得到的切空间残差，边只由文本建立。
- `TRY-046`：若过平滑，加入根节点残差和单层传播限制。
- `TRY-047`：若错误邻居干扰，训练可关闭的边Gate并限制top-k。
- `TRY-048`：若跨seed不稳定，使用三种k的图重心；仍失败则停止SGT。

### MPR：多原型角色混合，TRY-049至TRY-052

- `TRY-049`：每类保留local/unique/overall三个完整原型，不提前平均。
- `TRY-050`：用图像CLS训练三原型稀疏路由，三组对全部200类生效。
- `TRY-051`：若单组塌缩，加入最小使用率与熵下界。
- `TRY-052`：若不稳定，改为top-2原型竞争；仍失败则停止MPR。

### PGO：梯度冲突优化，TRY-053至TRY-056

- `TRY-053`：分别计算pseudo-seen CE、pseudo-unseen CE和topology梯度，冲突时使用PCGrad投影。
- `TRY-054`：若U/S偏置仍大，使用约束优化，限制任一目标梯度被完全覆盖。
- `TRY-055`：若训练震荡，按梯度范数动态归一化三个目标。
- `TRY-056`：若跨seed不稳定，冻结目标权重并做三初始化；仍失败则停止PGO。

### 组合与链式消融，TRY-057至TRY-060

- `TRY-057`：TG-VPR only，统一当前代码与数据身份的module-off基线。
- `TRY-058`：TG-VPR + TST。
- `TRY-059`：TG-VPR + TST + 当前最佳新模块。
- `TRY-060`：TG-VPR + 最佳两个新模块；检查是否互补而非重复。

### 最终可靠性与规范化，TRY-061至TRY-064

- `TRY-061`：最终组合seed5。
- `TRY-062`：最终组合seed6。
- `TRY-063`：最终组合seed8。
- `TRY-064`：最终module-off、错误邻居/描述shuffle控制和计算开销记录；一个控制问题可包含多个Condition/RUN。

如果前述模块提前止损，未使用编号立即分配给新的框架假设，不能用参数网格凑满数量。到`TRY-050`时至少完成50组真实运行；达到78%后继续完成剩余消融、seed和可靠性实验，目标不提前结束。

## 统一晋级与止损

- 首次探索固定seed7；相对准确父条件最高seed`Delta H>=0.20`才优先作为核心创新。
- U或S任一下降超过2个百分点，或出现参数/分量饱和，必须先诊断再补救。
- 每个模块最多首次TRY加3次方法级补救；工程重跑不计补救，但必须留失败状态。
- 多seed只在seed7通过后执行。range用于稳定性判断，最高seed用于主成绩。
- promote后才创建正式Experiment目录、PARAMETER_MATRIX和实验HTML图；失败只保留清单行与Idea结论。
- 最终论文必须提供链式消融：TG-VPR、TG-VPR+TST、TG-VPR+TST+新创新，以及必要的module-off控制。
