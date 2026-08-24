# V2-INNOVATION-094 结果

状态：`rejected_not_two_seed_consistent`。

RUN-001（seed5）最高`U/S/H/ZS=76.848692/80.265528/78.519956/84.020644%`，selected iteration=`705`。相对稳定SNPS top-3提高H `0.053246`。

11个证据dropout计数为2566或2567，覆盖严格均衡；集中化schema分发与方法训练均有效。追加seed7。

模型SHA256：`a0570ae332c5181d8d08049b1357707e6ef26a102b1789ad18e8e436391898ba`；最后checkpoint SHA256：`564070b628581514be13b097d72c865a093c8920b106ba83a9a90a10661ba59e`。

RUN-002（seed7）最高`U/S/H/ZS=76.846421/80.017334/78.399828/83.920074%`，selected iteration=`705`，比同seed稳定SNPS top-3低`0.046272`。

EDPS2相对top-3的seed5/7增量为`+0.053246/-0.046272`，方向不一致。集中化重实现给出了有效方法结果，但证据dropout未跨seed成立，IDEA-127最终拒绝。

RUN-002模型SHA256：`0421d032f7aa79e89532a913a964b27b6f988e6f044162f5724c94e2b0f15dbd`；最后checkpoint SHA256：`f0f0001baee673e1b5335ecbda52f6100af8c024ba8d0f70a8c1c1c4456301c`。
