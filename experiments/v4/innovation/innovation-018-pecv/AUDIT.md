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

- Post-fix提交：`6043e1d7cc5c1af219a7dc0a952398faff27c91f`。
- 同Agent A/B并行复核5组P1均已关闭，未发现新P0/P1。
- 双方直接交换复核清单并完成质询互认：`P0=0/P1=0`。
- 共同结论：**代码单轮双Agent对抗审核通过**。

剩余P2：失败中断会留下空输出目录；shuffle只支持整体语义映射claim；图像频率训练与macro评估可能偏移；Gate checkpoint只验证严格恢复输出，不承诺任意中间步续训等值；TF32需在micro记录。

## GPU micro与真实Gate

- micro commit/config：`6043e1d` / `10ddca9d...`。
- micro：loss `0.391060`，finite gradient、optimizer step、Parent receipt exact、checkpoint roundtrip全部通过；RTX 4090峰值显存`115,508,224` bytes。
- TF32：CUDA matmul=`false`，cuDNN=`true`；本Gate不使用卷积。
- Gate训练：固定1000 updates完成，输出metrics/evidence/checkpoint三文件；代码树clean，训练结束后GPU释放。
