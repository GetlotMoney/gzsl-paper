# V2-INNOVATION-058 结果

状态：`testing_seed7_reliability`。

RUN-001使用seed5父链，train-only门槛由`416`个“错误且top2同族”样本的margin中位数固定为`0.542624`。最高`U/S/H/ZS=76.647568/80.107862/78.339523/83.888441%`，相对SDCR H提高`0.019013`个百分点。

best beta=`-2.033723`未饱和；seen/unseen GZSL平均gate=`0.152989/0.182175`，门控真实生效。增益很小且U/ZS略降，暂只保留弱正信号，追加seed7可靠性。

模型SHA256：`ae6c61ac0186184a6e2477065d99ad9c7f19cb49f805b31ebba57e007f27eaed`；最后checkpoint SHA256：`1e4e80528962135ed7fba0f10001f55d4d26e7c3f9622c990c5797304041093c`。
