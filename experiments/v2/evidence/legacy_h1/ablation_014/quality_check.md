# 质量检查

- 四个RUN均在服务器独立worktree的同一commit执行。
- seed、数据SHA、50轮训练、损失、scheduler和评估口径一致。
- A↔C、B↔C、C↔D为预注册单变量比较。
- 每个RUN均保存training.log、model_best.pth和metrics.json。
- official test在epoch 50 checkpoint保存后读取。
- 结果标记为component ablation与not confirmation evidence。

结论：`pass`。
