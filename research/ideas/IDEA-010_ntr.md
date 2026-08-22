# IDEA-010：NTR邻域感知切空间路由

```yaml
idea_id: IDEA-010
source_type: local_model_analysis
evidence_refs:
  - V2-INNOVATION-002
  - model/innovations/elpt.py
base_commit: 0b919b14f052ec5e3f99378383e94053a2cf45ae
problem: TST gate只接收top-5相似度的均值与最大值，无法区分单一近邻和多个近邻形成的不同语义局部结构。
hypothesis: 输入完整top-5邻域相似度分布，可让gate更准确地决定每个类的切空间迁移步长并提高H。
core_change: gate输入由4维摘要扩展为8维：Mean8-Value余弦、位移、top-5均值及完整5个邻居相似度；其余TST不变。
success_condition: seed7相对TG-VPR+TST的DeltaH不低于0.05个百分点，U和S各自下降不超过2个百分点，并保持TST步长与角位移安全门槛。
failure_condition: 首次TRY和最多3次方法级补救后仍不满足成功条件。
status: revised
paper_core_innovation: false
parent_condition: V2-INNOVATION-002 / TG-VPR + TST
current_attempt: none
last_attempt: V2-TRY-036
last_decision: keep_as_current_best_observation
```

NTR只改变训练式类别gate的语义几何输入；true-unseen图像不进入梯度。改回4维摘要时严格回到TST。

## V2-TRY-028结果

seed7得到`U=74.319029%`、`S=80.068129%`、`H=77.086536%`、`ZS=81.430238%`，相对TST四项全部提高，`ΔH=+0.101991`；相对原V2 `ΔH=+3.063354`，通过项目seed7目标。结构现已冻结，下一步运行seed5/6/8独立三折训练。

## 四seed诊断

相对TST的seed 5/6/7/8 `ΔH=-0.114005 / +0.055576 / +0.101991 / -0.001984`，只有2/4为正，触发跨seed不稳定。补救1不再用8维gate整体替换TST，而是冻结4维TST gate，只训练由完整top-5邻域驱动、范围±0.1且零初始化的残差步长。

## V2-TRY-032结果

残差NTR的seed7得到`H=77.051348%`，相对TST `ΔH=+0.066803`、ZS提高`0.040007`，通过门槛。结构现已冻结，下一步在seed5/6/8加载各自TST gate并只训练邻域残差。

## 残差多seed诊断

残差条件的seed 5/6/7/8 `ΔH=-0.209205 / -0.085143 / +0.066803 / +0.029781`，仍只有2/4为正。补救2将完整top-5向量压缩为一个邻域标准差，与原4维摘要组成5维低复杂度gate；若seed7不通过则停止。

## V2-TRY-036结果与止损

5维离散度条件得到`H=76.806996%`、相对TST `ΔH=-0.177549`，seed7即未通过。NTR已验证完整向量、冻结残差和低维离散度三种结构，均未形成稳定4/4增益，IDEA-010标记`rejected`并提前止损。

保留观察：直接8维NTR的seed7为`77.086536%`，四seedH mean约`76.876640%`，两项数值目标均达到；但相对TST只有seed6/7为正，seed5/8为负，因此不能作为稳定第3创新晋级。

## owner新成绩口径下的复核

owner后续规定主成绩取最高seed，mean/range只判断偶然性；`range<=1.0`个百分点时可采用最高值。直接8维NTR四seedH范围约`0.5432`，因此`V2-TRY-028 / seed7 / H=77.086536%`现作为当前最佳框架观察保留。由于其相对TST最高增益仅`0.101991`个百分点，仍不作为论文核心第三创新，后续由更强的新框架替代或增强。

新增seed9得到`H=76.795441%`，相对seed9 TST提高`0.096995`，U/S/ZS也均提高。直接NTR五seed最高仍为seed7 `77.086536%`，min/max/range约为`76.543327 / 77.086536 / 0.543209`，继续满足owner的差距不大标准。
