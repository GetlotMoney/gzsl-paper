# V5-TUNE-005 结果

TG-VPR-H1 在 seeds 5/6/7/8 下的 H 为 `73.709453 / 73.864640 / 74.019664 / 73.759312`。

当前 `best_observed` 是 seed 7：

| Seed | U | S | H | ZS | Epoch |
|---:|---:|---:|---:|---:|---:|
| 7 | 72.690260 | 75.398600 | 74.019664 | 81.534684 | 50 |

四 seed 汇总：

```text
H mean  = 73.838267
H min   = 73.709453
H max   = 74.019664
H range = 0.310211
```

四个 seed 的 H 均高于 X2 历史单次 `H=73.523304`。该批只回答多 seed 稳定性，明确标记为 `not_confirmation_evidence=true`；不能只选 seed 7 写成 confirmed、promotion 或正式主线结果。
