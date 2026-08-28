# V3-TRY-040/041 GTD从头训练确认

- Owner确认base：`b548bfc08acfcf029eaf0b3017b1e5b722aec661`。
- RUN代码提交：`bb7d1610a13970347b7e4cfb05fa1db765386d7e`。
- TRY-040 config SHA：`a041e26cded05522ad38faa96b78852788eeb0566f938339a42468f4436bad29`。
- TRY-041 config SHA：`e39981f35198a82b8d4d119100251b79b2b0f4e31cb8f9ae4a21f67780405f8b`。
- 本机资产manifest SHA：`3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6`，从lab4090复制必需的全局CLS、标签和八角色文本张量到仓库外本地Warehouse。
- 初始化：两个条件均不加载TG checkpoint；在seed7下构造同一TG与零初始化GTD scaffold，记录`initial_tg_state_sha256`。
- 唯一条件差：TRY-040的`gate_loss_weight=0`并用TG父原型评估；TRY-041的`gate_loss_weight=1`并用GTD迁移后原型评估。其余训练配置完全相同。
- 训练：batch50、固定150名义epoch/21,171 updates、每141步及尾点official评估，共152点；CLIP图像/文本特征冻结。
- 证明目标：两RUN的初始TG SHA、batch序列和TG父梯度必须一致；比较global-best Full指标，并在TRY-041同checkpoint报告GTD-off四指标。
- 本地测试：专项`11 passed`，全仓`535 passed / 2既有warnings / 3 subtests passed`。
- 审核状态：侧对话不能调用独立Agent，本次只允许本机诊断性复现；结果不得冒充已通过项目两轮审核的正式论文证据。

