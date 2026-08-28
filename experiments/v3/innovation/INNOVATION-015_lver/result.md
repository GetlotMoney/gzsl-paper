# V3-INNOVATION-015 结果

状态：`completed_rejected`。

V3-TRY-048是fresh TG+GTD共同父条件；V3-TRY-049在完全相同TG/GTD初始化、主batch和21171 updates时间线上从update 1同步训练LVER。所有模块禁止加载训练checkpoint。

| RUN | U | S | H | ZS | ΔH_add | ΔH_remove | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| V3-TRY-048 TG+GTD | 79.758894 | 76.615125 | 78.155408 | 85.828280 | — | +1.506388（GTD自身） | 父条件 |
| V3-TRY-049 +LVER | 79.758894 | 76.615125 | 78.155408 | 85.828280 | 0.000000 | 0.000000 | drop |

LVER的best checkpoint与匹配父条件完全相同，未形成任何独立贡献。两条RUN均满足`loaded_training_checkpoints=[] / completed_fixed_150 / 21171 updates / 152点评估`，结果来自各自同一个best-H checkpoint。
