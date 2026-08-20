# V1-CONFIRM-001 结果

状态：`pending_RUN-002`。

RUN-001 在模型初始化时因类别校验 device mismatch 失败，尚未发生训练 step，失败证据保留。RUN-002 使用最小修复提交和独立输出目录重跑；完成后从 `PARAMETER_MATRIX.csv` 汇总真实 U/S/H/ZS、best epoch、数据指纹、日志 URI 和模型 URI。
