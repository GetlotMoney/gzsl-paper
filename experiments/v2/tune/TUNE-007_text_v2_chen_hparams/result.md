# V2-TUNE-007 结果

状态：topology_coarse_completed_paused_by_owner。

| 数据集 | topology | H | best stage | 模块非零 | 当前判定 |
|---|---:|---:|---|---|---|
| CUB | 0 | 77.362837 | TRANSFER_CCGR | 是 | 粗搜选择；与0.1差<0.1且参数更小 |
| CUB | 0.03 | 77.352014 | TRANSFER_CCGR | 是 | 不选 |
| CUB | 0.1 | 77.458777 | TRANSFER_CCGR | 是 | reference |
| CUB | 0.2 | 76.819857 | TRANSFER_CCGR | 是 | 不选 |
| AWA2 | 0 | 95.497853 | TG_ONLY | 否 | 淘汰 |
| AWA2 | 0.03 | 96.145753 | TG_ONLY | 否 | 淘汰 |
| AWA2 | 0.1 | 95.666877 | TG_ONLY | 否 | 淘汰 |
| AWA2 | 0.2 | 95.012643 | TG_ONLY | 否 | 淘汰 |
| SUN | 0 | 70.040488 | JOINT_FINETUNE | 是 | 暂时领先 |
| SUN | 0.03 | 69.316184 | JOINT_FINETUNE | 是 | 不选 |
| SUN | 0.1 | 68.329691 | JOINT_FINETUNE | 是 | reference |
| SUN | 0.2 | 67.377600 | TRANSFER_CCGR | 是 | 不选 |

CUB topology轴选择0并进入Transport粗搜。AWA2四个条件的全局best均位于TG-only，按预注册非零模块门槛关闭Stagewise细搜。SUN选择topology=0。Owner要求当前粗搜完成后暂停；没有启动Transport或后续轴。
