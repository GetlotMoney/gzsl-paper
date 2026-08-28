# V3-CONFIRM-014 审查状态

- 最终RUN code commit：`13bd1ccb513710ce798fbaa7147af447d43b0b36`
- MMT公式来源：`e6cae35a759bf5e40a5900c30a4cb0330fa1f06e`
- BD公式来源：现有BD-TST当前公式模块（语义内容一致）
- 本地专项：`completed teacher checkpoint validate/load/restore = 1 passed`
- 本地全量：`547 passed, 1 CUDA test skipped locally, 3 subtests passed`
- 真实资产manifest：`3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6`
- 服务器CUDA micro-batch：四种objective均已复用`4e2195e`共享证据；训练数值路径在`13bd1cc`未改变。
- 服务器CUDA RNG专项：四条件update 1的CUDA RNG、父logits和TG梯度严格匹配。
- 旧audit smoke：`V3-TRY-042/043/044/045 = invalid_rng_mismatch`；不得作为本修复commit证据。
- 旧正式RUN：`V3-TRY-043`完成21171 updates后因teacher refresh错误断言未发布；`V3-TRY-044`随即停止。两者保留为`invalid-teacher-count-contract`。
- Round 1：`13bd1cc / P0=0, P1=0, P2=0 / 第1轮通过`
- Round 2：`13bd1cc / 服务器HEAD准确且clean / P0=0, P1=0 / 无P0/P1，第2轮通过`
- 审核结论：`approved; V3-TRY-042–045必须全部绑定13bd1cc重新运行`
- 正式RUN闭环：四条RUN均已绑定`13bd1cc`完成；`loaded_training_checkpoints=[]`、`stop_reason=completed_fixed_150`、`total_updates=21171`、`history_length=152`，结果见`result.md`。

最终修复令teacher刷新严格覆盖`1,142,...,21010,21151`共151个区间起点，并允许`update=21171`的完成态checkpoint只恢复最终产物；缺失任一刷新项会被拒绝。初始化、forward、loss、梯度、配置、资产和summarizer均未改变。
