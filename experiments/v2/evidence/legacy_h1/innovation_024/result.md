# V5-INNOVATION-024 结果

TG-VPR-H1 已从历史多条件诊断入口无损提取为独立模型和训练入口。

| Seed | U | S | H | ZS | Epoch |
|---:|---:|---:|---:|---:|---:|
| 7 | 72.690260 | 75.398600 | 74.019664 | 81.534684 | 50 |

迁移等价检查：

```text
模型state keys：完全一致（13项）
模型state最大差：0
50轮history：完全一致
三组权重：完全一致
U/S/H/ZS：完全一致
```

本次只证明独立代码无损迁移，不改变多seed结果的 `not_confirmation_evidence` 边界，也不自动promotion。

证据：`warehouse://runs/tg-vpr-h1-standalone-r2-20260821/RUN-001`。

```text
metrics.json SHA-256: 9c2219c8fc71d3dfd39834b93351873a683098da1b81f5a37c3e166f97d082f5
model_best.pth SHA-256: e723fe45b22ef2724e82a42609a2e03830e10472f7971f053cb4880ee53f7436
training.log SHA-256: e5e524fd76fffd2cd5e7bdbee6200b01388d3685635f287eabb5b90cdb1f2817
```
