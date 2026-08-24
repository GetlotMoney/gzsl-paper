# V2-INNOVATION-095 结果

状态：`testing_seed7_required`。

研究问题：稳定SNPS top-3权重作为阶段一，阶段二用EDPS的循环证据屏蔽继续训练，能否保留父模型并降低从零训练的跨seed波动。

成功条件：seed5 H超过SNPS父条件`78.466710%`；若成立再运行seed7。失败时保留真实结果并停止该分阶段补救。

RUN-001（seed5）得到`U/S/H/ZS=76.883745/80.285692/78.547901/84.055108%`，selected iteration=`282`。相对SNPS父模型H提高`0.081191`，U基本不变、S提高`0.168848`、ZS下降`0.066102`。

阶段二后期H逐步下降，说明有效区间是从稳定父权重出发的短程证据稳健化，而不是无限继续训练。按门槛追加seed7验证，当前不宣称稳定创新。

RUN-002预注册：使用seed7稳定SNPS top-3父checkpoint，只改变对应训练随机种子为7，其余分阶段公式和Chen-style评估频率不变。

模型SHA256：`c76f88eaeae02adf8b3de83753b0179103f4e4c2ba109fc8396dc674d77ab529`；最后checkpoint SHA256：`31f64c75a9e06ce43160a9ea0496e0831382e6c7a233a82848972deb0cd578f8`。
