# V4-TRY-020代码单轮双Agent对抗审核

## 冻结前共享证据

- 准确父提交：`52088f69d7ac4e574e7b63c28b21ac0da7789933`。
- 专项测试：`tests/test_pecv_gtd.py`，`5 passed`。
- GTD相关回归：合计`18 passed`；Windows Python 3.14运行时打印过非阻断access-violation诊断，但pytest退出成功。
- 完整测试：`542 passed, 2 warnings, 3 subtests passed`。
- 两份配置固定200名义epoch、28,228 updates、无TG/GTD/PECV checkpoint。
- 服务器micro和正式输出：审核前均不存在。

## 审查矩阵

- 公式：反对称纠错、Top-5零和聚合、truth-injected训练与部署同forward。
- 训练：TG/GTD/PECV三组参数、同步update 1、LR warmup、loss与batch轨迹。
- 评估：200类U/S/H、50类ZS、官方test选择、同checkpoint Off、匹配Parent。
- 数据：仅seen图像进梯度，禁用人工属性/部位/框/专家残差。
- 运行：配置/资产SHA、checkpoint恢复、202评估点、200 teacher refresh、完成态。
- 输出：Full/Parent同源、initial state、loaded checkpoint、metrics/history/checkpoint合同。

审查对象commit、双方独立发现、直接交叉质询、集中修复和最终共同结论将在代码冻结后回填。
