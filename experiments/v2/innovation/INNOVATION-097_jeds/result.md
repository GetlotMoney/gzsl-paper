# V2-INNOVATION-097 结果

状态：`planned`。

研究问题：S-EDPS每batch只屏蔽一个证据，mask与随机batch的偶然配对是否限制了泛化。

唯一改动：每个seen batch构造11种单证据缺失视图，平均11个pair CE再更新一次；允许各视图修正不同，不使用CEPS一致性loss。seed5须超过S-EDPS`78.572828%`才追加seed7。
