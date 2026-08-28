# IDEA-155：Fresh一段式有效模块复验

- 状态：`revised`（控制复验已完成；GTD、MMT成立，BD失败）
- source_type：`owner_protocol_correction + prior_experiment_diagnostic`
- 性质：控制性复验，不是新的论文创新点。
- 问题：既有GTD、MMT、BD成绩加载了训练好的TG checkpoint，不能回答“所有启用模块是否从update1开始在唯一一段训练中同步学习”。
- 核心假设：若候选机制本身有效，它在seed7 fresh TG、相同主batch、相同TG学习率和150轮预算下，应同时优于匹配TG-only控制，并在同一个best checkpoint关闭自身后显示独立增益。
- 唯一控制改动：禁止加载任何CUB训练所得TG/GTD/MMT/BD权重；四条件统一由CPU构造fresh PaperV2 TG，再在`fork_rng`内初始化候选Gate，保持TG/dropout RNG与主batch初态一致。
- 条件：TRY042 TG-only；TRY043 TG+GTD；TRY044 TG+MMT；TRY045 TG+BD。
- 成立门槛：候选best H减TRY042 best H至少1.0，且候选best checkpoint的Full H减Module-Off H至少1.0，同时`|U-S|<8`。两项均至少0.8但不足1.0只记weak。
- 训练：CUB trainval 7057张、seed7、batch50、150名义epoch、21171 updates、每141步official test并在21171补评估；TG LR恒定1e-4；Gate从update1同步训练，5轮warmup后cosine。
- 披露：`test_used_for_selection=true / unseen_images_used_for_gradient=false / strict_blind_claim=false`。
- 失败：任一独立增益不足0.8或`|U-S|>=8`即drop；不以总体H替代模块独立效果。
- Experiment：`experiments/v3/confirmation/CONFIRM-014_fresh_effective_modules/`。
- 代码commit：`13bd1ccb513710ce798fbaa7147af447d43b0b36`。
- 结果：TRY042 TG `H=76.649020`；TRY043 GTD `H=78.155408`，相对TG `+1.506388`、同checkpoint关闭 `+1.506388`，保留为第一创新候选；TRY044 MMT `H=77.746215`，相对TG `+1.097195`、同checkpoint关闭 `+1.379207`，但与GTD同属unseen原型迁移，只保留为替代设计；TRY045 BD `H=74.786137`，相对TG `-1.862884`，淘汰。
- 证据状态：四条正式RUN均完成；`loaded_training_checkpoints=[]`、`stop_reason=completed_fixed_150`、`total_updates=21171`、`history_length=152`，且U/S/H/ZS与Full/Off均来自各自同一个best-H checkpoint。当前仅seed7，不能声称跨seed稳定性。
