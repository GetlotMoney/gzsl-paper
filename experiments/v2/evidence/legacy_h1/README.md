# H1旧实验直接证据

owner于2026-08-22明确授权，将H1相关旧实验的轻量证据直接迁入V2。旧实验ID只作为`legacy_ref`，不改变当前仓库的新编号体系。

迁入证据：

- `innovation_024`：独立代码与来源实现严格等价；
- `ablation_014`：Value路径是主要机制，三组结构有小幅收益，可学习组权重无贡献；
- `tune_005`：早期四seed稳定性记录；
- `tune_006`：最终参数收口和固定等权四seed结果。

核心结果：

```text
三组结构相对单组Value：+0.131688 H
Value路径相对无Value：+6.376252 H
可学习组权重相对固定1/3：-0.003517 H
最终四seed H mean/min/max/range：73.853094 / 73.709453 / 74.023182 / 0.313729
```

这些证据支持TG-VPR-H1的机制选择，但使用official test进行结构与参数收口，明确不是blind-test，也不替代当前仓库`V2-CONFIRM-001`正式单seed基线。

逐文件身份见`FILE_MANIFEST.csv`，迁移边界见`MIGRATION.yaml`。

