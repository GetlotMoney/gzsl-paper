# 项目状态

## 当前版本

```yaml
repository: GetlotMoney/gzsl-paper
frameworks:
  - id: FRAMEWORK-V1
    branch: framework/v1
    tag: v1
    status: baseline_completed_with_runtime_device_fix
  - id: FRAMEWORK-V2
    branch: framework/v2
    tag: v2
    status: baseline_completed_single_seed
evaluation_protocol: test_selected_inductive_gzsl
paper_primary_framework: FRAMEWORK-V2
paper_baseline_H: 74.023182
paper_target_H: 78.0
target_supported_innovations: 3
supported_innovations: 2
current_seed7_H: 76.984545
current_multiseed_mean_H: 76.866245
current_best_observation_H: 77.384331
current_best_observation_seed: 7
completed_try_count: 57
minimum_required_try_count: 50
```

V1 来源于 `model/v5-template-v2@fb4b29b04087640890a532f105cb527d3a8c461b` 的必要运行代码，旧仓库历史、旧实验和旧账本没有迁入。

## FRAMEWORK-V2

owner已将来源身份`INNOVATION-MODULE-1 / TG-VPR-H1`提升为独立正式框架`FRAMEWORK-V2`。V2使用独立代码、配置和训练入口，不接入`FRAMEWORK-V1`。首个当前仓库正式基线已由`V2-CONFIRM-001 / RUN-001`完成：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`。

owner已授权直接迁移H1旧实验的轻量证据。组件消融、多seed和参数收口证据位于`experiments/v2/evidence/legacy_h1/`；`IDEA-001 / TG-VPR-H1`现为论文核心创新1，状态`supported`。

## 当前待办

owner已选择`FRAMEWORK-V2`作为论文主框架。V2当前正式单seed基线为`H=74.023182%`，新的三个百分点目标为`H >= 77.023182%`。

固定10%保守unseen迁移在四seed均提升H，但它只在测试时生效，现降级为`test_time_observation`，不计入论文核心创新。

训练式ELPT已完成`V2-TRY-006`及全部3次方法级补救。最佳H达到`76.803085%`，但首次TRY的gate均值超过预注册上限；三个补救又持续出现gate饱和或S下降超过2个百分点，因此`IDEA-002`已标记`rejected`并强制止损。没有建立`V2-INNOVATION-002`。

`IDEA-003 / ICGR`已完成首次TRY与两次适用的补救。原始路由和增加语义余弦输入均未提高H；均匀KL消除了权重塌缩，但最终仍为`H=73.976174%`、`ΔH=-0.047008`。只适用于跨seed不稳定的RESCUE-3前提不成立，因此该方向已提前止损并标记`rejected`。

当前仍只有`IDEA-001 / TG-VPR-H1`一个supported核心创新。下一步必须建立新的独立训练式候选，不能把ELPT或ICGR失败条件晋级为正式创新。

`IDEA-004 / ACGR`使U和ZS出现正向信号，但H未提升；一次保守幅度补救仍失败且发生组权重塌缩，现已标记`rejected`。下一候选回到原型迁移主线，改用切空间方向迁移的新公式，不复用ELPT实验身份。

`IDEA-005 / TST`已在seed 5/6/7/8全部提高H，平均提升`3.013152`个百分点，候选H mean=`76.866245%`，已超过四seed目标`76.853093%`，现为论文核心创新2。seed7为`76.984545%`，距离单点目标仍差`0.038637`个百分点；项目还缺第3个supported创新，因此整体工作未完成。

TST之后已依次止损EPC、CATA、SPA、PURL和NTR。NTR直接8维条件曾达到seed7 `H=77.086536%`、四seedH mean约`76.876640%`，但相对TST仅2/4 seed为正，未按稳定创新晋级。当前正式状态仍是2个supported创新；第3创新与完整三创新组合尚未完成。

owner已更新成绩口径：主结果报告最高seed，mean/range只用于判断偶然性；`range<=1.0`个百分点时可以最高seed作为主成绩。按此口径，当前最佳观察是`V2-TRY-028 / seed7 / H=77.086536%`，四seed范围约`0.5432`，可作为当前最佳框架参考。但NTR相对TST的最高增益只有`0.101991`个百分点，未达到新核心创新`0.20`个百分点门槛，因此继续搜索替代或增强模块。

长期计划已完成第50个有效实验：`V2-TRY-050 / TG-VPR seed9`得到`H=73.478685%`。达到50组不结束目标，下一步继续运行seed9对应TST与NTR，并推进新的框架候选。

seed9后续结果：TST `H=76.698446%`，NTR `H=76.795441%`。NTR相对TST四项均提高，`Delta H=+0.096995`；五seed最高仍为seed7 `77.086536%`，范围约`0.543209`。当前累计52组有效实验，稳定78%+尚未实现。

当前累计55组有效实验。BMR、DPT、SGT、MPR、PGO与SVPG均已按真实失败模式止损；其中SVPG再次证明直接把seen视觉映射施加给unseen会造成严重联合竞争偏置。稳定78%+仍未实现，下一主线转向正交残差与类别条件生成。

当前累计57组有效实验。正交残差主子空间与补空间均被训练关闭，ORT已止损；全局共享SVPG和低秩ORT共同证明seen视觉偏置不能整体迁移给unseen。下一主线必须使用类别条件机制，稳定78%+仍未实现。

CCGR类别条件文本几何生成在seed7得到`H=77.100834%`，比NTR提高`0.014298`，成为当前最高观察，但未达到核心创新门槛。下一条件改用pseudo-unseen episode直接训练CCGR Gate。

episodic CCGR进一步达到`H=77.237120%`，相对NTR提高`0.150584`，成为当前最高观察；仍未达到78%，且U/S偏向需要继续补救。

unseen平衡CCGR进一步达到`H=77.384331%`，相对NTR提高`0.297795`且U/S/ZS全部提高，首次达到新核心创新增益门槛。当前仍需完成幅度非饱和补救、多seed和正式消融，78%目标尚未达到。

新的长期目标是稳定达到最高seed `H>=78.0%`、形成3个可解释且有消融支撑的创新，并累计完成至少50组真实实验。执行计划见[`docs/LONG_HORIZON_EXPERIMENT_PLAN.md`](LONG_HORIZON_EXPERIMENT_PLAN.md)。

完整执行顺序和完成条件见[`docs/PROJECT_CHECKLIST.md`](PROJECT_CHECKLIST.md)。
