# V2-TUNE-001 结果

状态：`topology_0.03_rejected`。

RUN-001为无专家属性CLIP/GPT文本路线，RUN-002在同一视觉特征和训练协议上增加CUB 312维专家属性残差。

两条RUN只使用xlsa17开发训练/validation，均未加载official test。

| Condition | U_val | S_val | H_val | ZS_val | selected epoch |
|---|---:|---:|---:|---:|---:|
| RUN-001 无专家 | 76.424742 | 76.521248 | 76.472964 | 79.934835 | 24 |
| RUN-002 专家312维属性 | 77.498114 | 77.537417 | 77.517761 | 81.446886 | 25 |
| 专家 - 无专家 | +1.073372 | +1.016170 | **+1.044797** | +1.512051 | - |

两个条件都在每轮完整且唯一遍历3,724张开发梯度图像；978张validation-seen和2,355张validation-unseen图像不进入梯度。专家属性残差最终为`0.192030`，未达到`0.5`上限。

这只是validation选模结果，不是official test成绩。当前遗留CLIP缓存来源不完整，两个条件均保持`final_test_eligible=false`。

模型SHA：

- RUN-001：`d99156ff078c0584f80497be7a86bef06a76180c6fc499df4135f8cc7d6c97c8`
- RUN-002：`e3bc0bdc8596d6c09106939784c147bd8871915e52838990d1d0778c40e01c53`

下一对RUN-003/004仅把`topology_weight`从`0.1`降为`0.03`，根据validation检查拓扑约束是否过强。

RUN-003无专家`H_val=76.397660`，相对RUN-001下降`0.075304`；RUN-004专家`H_val=77.477349`，相对RUN-002下降`0.040412`。两个条件都不保留，`topology_weight=0.1`继续作为当前validation最优。
