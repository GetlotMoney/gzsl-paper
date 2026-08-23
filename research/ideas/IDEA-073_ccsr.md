# IDEA-073：Class-Conditioned Conservative Sentence Routing

status: testing
problem: CASR为所有200类共享同一组八句权重，但某句话相对类名提供多少独立语义会随类别变化。
hypothesis: 固定CASR全局权重，以每类每句相对类名的标准化独立度为特征，只学习一个共享斜率，可产生受控类别差异并超过CASR。
evidence_refs: IDEA-072 CASR两seed可靠；IDEA-071自由路由会塌缩；因此只允许一个共享斜率而不训练200×8参数。
base_commit: abd371fde89ca3cda1bf32181e7d26ce075a0a0a
core_change: 固定CASR全局句权重和beta，按句子-类名独立度训练一个有界共享delta；delta=0严格复现CASR。
success_condition: H大于CASR最高78.285719，U和S任一项下降不超过2个百分点，delta不饱和且类间权重variation大于0.005。
failure_condition: H不超过CASR、delta饱和或类间权重近常数。
experiment: V2-INNOVATION-039
