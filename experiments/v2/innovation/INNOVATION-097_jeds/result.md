# V2-INNOVATION-097 结果

状态：`rejected_no_gain_extra_compute`。

研究问题：S-EDPS每batch只屏蔽一个证据，mask与随机batch的偶然配对是否限制了泛化。

唯一改动：每个seen batch构造11种单证据缺失视图，平均11个pair CE再更新一次；允许各视图修正不同，不使用CEPS一致性loss。seed5须超过S-EDPS`78.572828%`才追加seed7。

RUN-001得到`U/S/H/ZS=76.982599/80.230141/78.572828/84.121776%`、selected iteration=`2820`，与S-EDPS逐项相同。11个维度各完成28,228次缺失覆盖，训练机制真实执行，但没有带来分类变化。

JEDS增加约11倍selector视图计算却无增益，说明S-EDPS循环单mask已经足够消除主要证据依赖；IDEA-130拒绝，不追加seed7或mask组合补救。模型SHA256：`4b8084b73e35858e0af70ce22d74560e93a14f6d2a52d12a4188f6dd776c0734`；最后checkpoint SHA256：`41cc0ff66ca35c949e8f05d29c381f5e3e250bc7a99ee38133da3d74ca171d7a`。
