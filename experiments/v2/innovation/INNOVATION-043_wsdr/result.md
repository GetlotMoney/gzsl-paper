# V2-INNOVATION-043 结果

状态：`rejected_below_sdcr`。

RUN-001得到`U/S/H/ZS=76.679766/79.959893/78.285486/83.920646%`，best位于iteration `18189`。相对CASR seed5父模型`H=78.276696%`仅提高`0.008790`个百分点，低于SDCR最高`78.320510%`，未通过预注册成功门槛。

八句权重`std/min/max=0.055120/0.046175/0.234550`，两个候选mask累计覆盖均衡，说明失败不是权重塌缩，而是“每批只优化两个候选中的更坏者”没有带来比随机单句dropout更好的泛化。IDEA-077拒绝；最坏候选dropout轴止损，不追加相同方向参数搜索。

该RUN使用7,057张seen训练图像，真实unseen图像未进入梯度；official test评估`202`次并用于选择iteration，明确`test_used_for_selection=true`，不作blind-test声明。模型SHA256：`136d78bee10ea97500dffe6ecc3786de784798f9f7bbbe76ee8a8b14fb3ea508`；最后checkpoint SHA256：`85aab777097375446055cec2f5fdb406a53175c9dea4525f84c350628f53a614`。
