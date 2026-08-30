# PECV代码单轮双Agent对抗审核

## 冻结身份与共享证据

- 初审提交：`fb133fec4ad1506461706555b702792459c3282b`
- 准确父提交：`52088f69d7ac4e574e7b63c28b21ac0da7789933`
- 专项测试：`3 passed`
- 初审前完整测试：`540 passed, 2 warnings, 3 subtests passed`
- 服务器训练与micro：初审期间均未启动。

## 两名Agent独立发现与直接交叉质询

Agent A与Agent B先独立审完，再直接交换完整清单并逐项质询，双方共同互认以下去重结果：`P0=0`，`P1=5组`。

1. 单pair训练与Top-5部署聚合不一致。
2. `module_off_exact`是同一函数自比，恒真。
3. Gate遗漏Idea预注册的`net_correction>0`。
4. expected-config、Parent上游、候选范围/唯一性、轴和finite合同不足。
5. strict determinism、checkpoint严格roundtrip、evidence SHA与环境指纹不足。

P2：语义shuffle只能支持整体语义映射贡献；class-frequency训练与macro评估有偏移风险；tie规则、原子输出及micro/macro coverage需明确。

双方结论：初审提交不得签字、不得micro；先做一次集中修复。

## 集中修复

- 训练改为每图`truth + Parent最强4个错误seen类`，真实调用部署同一个Top-5十pair零和forward并优化候选内交叉熵。
- Parent关闭路径改为与冻结candidate receipt的Top-1和Top-5比较。
- Gate加入`net_correction>0`。
- 加入expected config SHA、Parent上游身份、100/50轴、hard negative、Top-5唯一性/range与finite检查。
- 开启strict determinism，原子保存可重建checkpoint和evidence，严格load roundtrip，并记录SHA与环境/GPU指纹。
- 区分micro/macro coverage，记录tie规则，并把shuffle claim限定为整体候选语义。

## Post-fix复核

待同两名Agent针对最终post-fix提交并行复核后回填。只有双方均报告`P0=0/P1=0`、完成直接互认并共同写出“代码单轮双Agent对抗审核通过”后，才能启动服务器micro与Gate训练。
