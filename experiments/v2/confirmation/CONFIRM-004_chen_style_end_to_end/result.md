# V2-CONFIRM-004 结果

状态：`completed_expert_pass_no_expert_fail`。

两条RUN均端到端联合更新TG-VPR、TST/NTR和CCGR；专家条件额外更新312维属性残差。模块不分阶段选最大，只根据完整模型official H保存`model_best.pth`。

协议固定为Chen-style：trainval训练、每步独立随机抽50张、28,228次更新、每141步official test、`test_used_for_selection=true`。

| Condition | U | S | H | ZS | best iteration | nominal epoch |
|---|---:|---:|---:|---:|---:|---:|
| RUN-001 无专家端到端 | 69.692755 | 81.027550 | 74.933940 | 80.256838 | 9165 | 65 |
| RUN-002 专家端到端 | 74.429244 | 82.228470 | **78.134714** | 85.708600 | 6768 | 48 |
| 专家 - 无专家 | +4.736489 | +1.200920 | **+3.200774** | +5.451763 | - | - |

两条RUN均完成28,228次更新和201个official评估点。首个batch中TG-VPR、transport、generator均有非零有限梯度；专家RUN的属性分支同样有非零梯度。模块未分阶段选模，U/S/H/ZS来自同一个完整模型best-H checkpoint。

无专家端到端未达到`H >= 77.023182`目标；专家端到端达到`H >= 78`目标。后续若补分阶段方案，必须新建Experiment并保留本结果作为端到端对照。

模型SHA：

- RUN-001：`0b05d2d90dc17ac7500613b1af3826204c066717536fc3ad8a74a30bd0a61761`
- RUN-002：`aeebaf4e15176ae132f3775393f997a31899b0e55565adc9738c63a1d6d6f8ad`
