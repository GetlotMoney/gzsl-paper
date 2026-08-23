# IDEA-079：Multi-Geometry Sentence Routing

status: testing
problem: SDCR仍让200类共享一套句权重；CCSR只用“句子相对类名独立度”一个量和一个斜率，表达不足且已失败。
hypothesis: 用4个纯文本几何量和一组跨类共享系数生成受限类别权重，可学习能从seen类别迁移到unseen类别的路由规则，并超过SDCR。
evidence_refs: IDEA-075 SDCR两seed可靠；IDEA-073证明单一独立度不足；IDEA-074证明直接用图像CLS的动态门控会退化，因此本次只用可跨类迁移的文本几何。
base_commit: 4e4734b0f0885d9cccfd7f99488653b4b3bd0419
core_change: 从SDCR完整推理权重起步，新增4个文本几何特征和4个共享系数，为每类生成受限八句权重；不使用图像条件和真实unseen图像梯度。
success_condition: H大于SDCR最高78.320510，U和S任一项下降不超过2个百分点，class variation大于0.001且最小权重大于0.01。
failure_condition: H不超过SDCR、class variation近零、权重塌缩或共享规则在unseen上方向相反。
experiment: V2-INNOVATION-045
