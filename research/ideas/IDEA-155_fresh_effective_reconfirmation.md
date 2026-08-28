# IDEA-155：Fresh一段式有效模块复验

idea_id: IDEA-155
source_type: owner_protocol_correction + prior_experiment_diagnostic
status: revised
base_commit: 8ea2329191e40d15fb1dcd5aef7fb757f58766a8
problem: 既有GTD、MMT、BD成绩加载了训练好的TG checkpoint，不能回答所有模块能否从update 1在唯一一段训练中同步学习。
hypothesis: 有效候选在seed7 fresh TG、相同主batch、相同TG学习率和150轮预算下，应同时优于TG-only匹配控制，并在同一best checkpoint关闭自身后显示独立增益。
core_change: 禁止加载任何CUB训练checkpoint；四条件统一fresh初始化并同步训练。
success_condition: 相对TRY042与同checkpoint关闭的H增益均至少1.0，且|U-S|<8。
failure_condition: 任一独立增益低于0.8或|U-S|>=8。
experiment: 645b609:experiments/v3/confirmation/CONFIRM-014_fresh_effective_modules/
result: GTD H=78.155408并通过双门；MMT H=77.746215但与GTD同属原型迁移替代；BD H=74.786137并失败。当前仅seed7。

