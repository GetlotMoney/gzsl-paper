# V2-INNOVATION-088 结果

状态：`rejected_not_two_seed_consistent`。

RUN-001（seed5）最高`U/S/H/ZS=76.917619/80.101538/78.477298/83.988434%`，selected iteration=`423`。相对稳定SNPS top-3提高H `0.010588`。

全部4458个pair保留，gamma=2 focal CE有限且无需大类别权重；增益较小，追加seed7。

模型SHA256：`7c485ed9581838656a5b587329b324ce3bb20e842b84398b3167e8f285a362df`；最后checkpoint SHA256：`e7c59432c31278762c4010160dd4adc8680859c30efb69e66f227ceede034bbd`。

RUN-002（seed7）最高`U/S/H/ZS=76.848710/79.939705/78.363739/83.953404%`，selected iteration=`141`，比同seed稳定SNPS top-3低`0.082361`。

FBPS相对top-3的seed5/7增量为`+0.010588/-0.082361`，方向不一致；IDEA-122拒绝，不替代普通pair CE。

RUN-002模型SHA256：`4c146b22d897935253b03062fb1110d18af9299326646c7074f88a0b76d18bfe`；最后checkpoint SHA256：`524dd565b6220a7464f2a2618f6b79a0ae7b992aac3ea56721ba14a6504aec84`。
