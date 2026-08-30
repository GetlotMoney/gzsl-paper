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

## 初审与直接交叉质询

- 初审提交：`cf427f0f02df5124a6d76e2f5d28ca9f61897b8b`。
- Agent A/B先独立完成全矩阵，再直接交换完整清单并逐项质询互认。
- 去重结论：`P0=0`，`P1=4组`。

1. Full额外PECV forward消耗TG dropout RNG，导致下一update起与Parent随机轨迹分叉。
2. same-RUN resume错误写入`loaded_training_checkpoints`且未冻结resume输入SHA。
3. 没有可续的主batch trajectory SHA，无法确认双RUN主batch完全一致。
4. 最终best只复算Full U/S/H/ZS，没有复算同checkpoint Off、delta和转移计数。

双方共同结论：初审身份不得micro或签字，先做一次集中修复。

## 集中修复

- 用`torch.random.fork_rng`隔离额外PECV训练forward，恢复后续Parent RNG流。
- `loaded_training_checkpoints`只描述初始化checkpoint并恒为`[]`；resume单列冻结path/SHA。
- 对每个update的50个主batch索引建立可恢复SHA256链，checkpoint与metrics均保存。
- 最佳状态严格复算Full、同checkpoint Off、四项delta和三split转移计数。

## Post-fix复核

待原Agent A/B对准确post-fix提交并行复核与直接互认；双方共同报告`P0=0/P1=0`前不得启动服务器micro。
