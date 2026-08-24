# V2-INNOVATION-095 结果

状态：`testing_seed7_required`。

研究问题：稳定SNPS top-3权重作为阶段一，阶段二用EDPS的循环证据屏蔽继续训练，能否保留父模型并降低从零训练的跨seed波动。

成功条件：seed5 H超过SNPS父条件`78.466710%`；若成立再运行seed7。失败时保留真实结果并停止该分阶段补救。

RUN-001（seed5）得到`U/S/H/ZS=76.883745/80.285692/78.547901/84.055108%`，selected iteration=`282`。相对SNPS父模型H提高`0.081191`，U基本不变、S提高`0.168848`、ZS下降`0.066102`。

阶段二后期H逐步下降，说明有效区间是从稳定父权重出发的短程证据稳健化，而不是无限继续训练。按门槛追加seed7验证，当前不宣称稳定创新。

RUN-002预注册：使用seed7稳定SNPS top-3父checkpoint，只改变对应训练随机种子为7，其余分阶段公式和Chen-style评估频率不变。

RUN-002（seed7）完整训练后best严格保持SNPS父模型`U/S/H/ZS=76.847547/80.112571/78.446100/83.987880%`，selected iteration=`-1`。首次S-EDPS的seed5/7增量为`+0.081191/+0.000000`，不能晋级稳定创新。

失败模式是阶段二`1e-3`学习率下seed7从第一次更新起即破坏父权重；RESCUE-1只把学习率降到`1e-4`，先在seed7复验，不增加新loss或输入。

RESCUE-1 RUN-003（seed7、lr=`1e-4`）得到`U/S/H/ZS=76.914215/80.112571/78.480820/84.021211%`，selected iteration=`282`，相对同seed父模型H提高`0.034720`。U和ZS同时提高，S不变；低学习率修复了seed7的即时退化。

必须用同一`1e-4`在seed5运行后才能判断跨seed稳定，不能把RUN-001的`1e-3` seed5与RUN-003拼接为同一条件。

RUN-003模型SHA256：`7d2c16704d999c4b650fd4c23f4911b5c3649da259bf34106d83b0e751e3b9fa`；最后checkpoint SHA256：`24c029ee8d5f83f7561ff381306a81688faeceeb579694f17e3142d969ede35e`。

RUN-002模型SHA256：`e04e81f0f7c7493d3a1d2526196e576ad372f2cff422db79fd6ac2426fd6b30c`；最后checkpoint SHA256：`9ce9e8a025c4502a809824ddc1b202075400f3d659f680424ec0b24b50d7f711`。

模型SHA256：`c76f88eaeae02adf8b3de83753b0179103f4e4c2ba109fc8396dc674d77ab529`；最后checkpoint SHA256：`31f64c75a9e06ce43160a9ea0496e0831382e6c7a233a82848972deb0cd578f8`。
