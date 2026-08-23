# IDEA-074：Image-Conditioned Conservative Sentence Routing

status: testing
problem: CASR对所有图像共享句权重；同一鸟类不同图像可见部位不同，固定句权重无法按当前视觉证据选择描述。
hypothesis: 固定CASR权重和beta，用零初始化CLS门控预测±0.5句logit残差，并以0.01 KL约束在父权重附近，可产生非塌缩图像动态路由并超过CASR。
evidence_refs: IDEA-072 CASR两seed可靠；IDEA-073类别固定路由无效，说明路由条件应来自当前图像而非静态类别几何；早期ICGR失败要求本次从CASR严格起步并限制残差。
base_commit: 943554e302f5a5338daa596e6fe666eaa580f6a8
core_change: 固定CASR全局权重和beta，增加零初始化`768→32→8`图像门控与0.01 KL，只训练门控。
success_condition: H大于CASR最高78.285719，U和S任一项下降不超过2个百分点，image variation大于0.005且min weight大于0.01。
failure_condition: H不超过CASR、动态性不足或权重塌缩。
experiment: V2-INNOVATION-040
