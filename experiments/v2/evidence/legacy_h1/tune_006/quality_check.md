# V5-TUNE-006 质量检查

结论：`pass`。

- 第一阶段7个新RUN和第二阶段3个新RUN全部完成，失败数为0。
- 两阶段均在独立clean server worktree启动，分别绑定唯一提交`4cf8baaf...`和`4c5c0910...`。
- 每个RUN具有独立目录、50轮history、`training.log`、配置快照、`model_best.pth`和`metrics.json`。
- 10个新RUN均记录完整U/S/H/ZS；ZS全部为`81.534684`。
- 第一阶段7个与第二阶段3个checkpoint SHA均已从文件重新计算并匹配metrics记录。
- 每阶段内部9项输入SHA完全一致，并与冻结config一致。
- 两个screen日志均未发现`Traceback/Error/Exception`，screen在队列完成后正常退出。
- RUN-004只复用`V5-ABLATION-014/RUN-003`，没有重复训练或覆盖历史目录。
- 结果明确标记test-exposed、`formal_evidence:false`和`not_confirmation_evidence:true`。
- 最终新增RUN数为10，满足最多13个的硬停止规则；实验已停止扩展。
