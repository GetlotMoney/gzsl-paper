# V2-INNOVATION-056 结果

状态：`rejected_uniform_sharpening`。

RUN-001训练alpha稳定为正约`0.24–0.28`，但H长期降到约`76.9%`；所有非零条件均低于父模型。best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`、selected iteration=`-1`、alpha=`0`。

统一放大族内中心化logit不会改变族内类别排序，只会放大已经排错的第一名。IDEA-090拒绝并关闭单标量族群缩放；下一方案必须使用类别对之间不同的语义权重，才可能改变排序。

模型SHA256：`33ceb7400885fa1621d55ef36e3154a086f32893b29888a339f018a178b01afc`；最后checkpoint SHA256：`d548ecfb07814b3179e63808eef8a1ad02e44ebe5f9fbb6a452d89f27c6d5605`。
