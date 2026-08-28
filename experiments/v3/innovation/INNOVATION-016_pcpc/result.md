# V3-INNOVATION-016 结果

状态：`completed_rejected`。

V3-TRY-048是fresh TG+GTD共同父条件；V3-TRY-050在完全相同TG/GTD初始化、主batch和21171 updates时间线上从update 1同步训练PCPC。所有模块禁止加载训练checkpoint。

| RUN | U | S | H | ZS | ΔH_add | ΔH_remove | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| V3-TRY-048 TG+GTD | 79.758894 | 76.615125 | 78.155408 | 85.828280 | — | +1.506388（GTD自身） | 父条件 |
| V3-TRY-050 +PCPC | 78.604460 | 77.336574 | 77.965362 | 85.342348 | -0.190046 | -0.190046 | drop |

PCPC略提高S但降低U、H和ZS；在自己的best checkpoint关闭PCPC后H反而回到78.155408。它没有提供独立细粒度增益。两条RUN均满足`loaded_training_checkpoints=[] / completed_fixed_150 / 21171 updates / 152点评估`，结果来自各自同一个best-H checkpoint。
