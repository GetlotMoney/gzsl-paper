# 实验与多 RUN 规范

## 核心对象

```text
Framework
└─ Experiment：一个研究问题
   ├─ Condition：基线、主方法、参数条件或控制条件
   │  └─ RUN：一套完整配置、一个 seed、一次执行
   └─ Condition
      └─ RUN
```

一个创新实验可以包含基线、主方法、多个参数、消融、多个 seed 和最终结果。不能把“一项创新实验”压缩成一个配置，也不能把每个参数值拆成新的创新编号。

## 实验目录

```text
INNOVATION-001_example/
├─ README.md
├─ EXPERIMENT.yaml
├─ module_source.md
├─ implementation.md
├─ framework_diagram.md
├─ configs/
│  ├─ RUN-001.yaml
│  └─ RUN-002.yaml
├─ PARAMETER_MATRIX.csv
├─ PARAMETER_MATRIX.md
├─ evidence/
│  ├─ RUN-001.md
│  └─ RUN-002.md
└─ result.md
```

## 参数矩阵

`PARAMETER_MATRIX.csv` 是唯一机器事实源。每个真实训练对应一行，至少包含：

```text
run_id,stage,condition,code_commit,config_ref,config_sha256,seed,
dataset_split,evaluation_protocol,status,U,S,H,ZS,best_epoch,
test_used_for_selection,log_uri,model_uri,decision
```

## Experiment 与 RUN 的边界

- 只改变 learning rate、rank、gate、loss 权重、epoch、seed 或预注册开关：同一 Experiment 的新 RUN。
- 改变模块公式、输入信息、forward、loss、seen/unseen 边界或评估语义：新建 Experiment。
- 小规模参数选择可以留在 Innovation；模块成立后的系统性超参数搜索进入 Tune。

## 推荐阶段

1. `baseline`：同 commit、同数据和同评估口径的基线。
2. `main`：创新模块默认条件。
3. `parameter`：少量预注册参数条件。
4. `control`：module-off、shuffle、wrong-role 等机制控制。
5. `repeat`：值得保留后再跑其他 seed。

项目允许根据 official test U/S/H/ZS 选择参数、epoch 和模型。每次 RUN 必须标记：

```yaml
evaluation_protocol: test_selected_inductive_gzsl
test_used_for_selection: true
unseen_images_used_for_gradient: false
```

这不是 blind-test 证据，任何论文数字或对外比较都必须如实说明。
