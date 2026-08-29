# V2-INNOVATION-075 结果

状态：`rejected_not_reproducible`。

首次启动在创建输出目录前因patch-free schema漏接线而`failed_pre_run`，没有开始训练且不计有效方法次数；commit `359bbc6`修复后使用同一未创建的RUN-001目录正式运行。

RUN-001（seed5）最高`U/S/H/ZS=76.949829/80.056471/78.472415/84.052283%`，selected iteration=`705`。相对稳定SNPS top-3提高H `0.005705`；第13维均值/标准差=`1.262952/1.144100`且selector权重=`0.042346`，特征有效但增益极小。

模型SHA256：`0680133380674c99b8b8beb768f56683e8458f700d87db6d369f78e5f3f428dc`；最后checkpoint SHA256：`8eabc875e9340c1d1ee7637741ebbbe63ba97c020c8b42f679cf41b81fb37d76`。

RUN-002（seed7）最高`U/S/H/ZS=76.748145/80.184466/78.428683/83.953416%`，selected iteration=`564`；比同seed稳定SNPS top-3低`0.017417`。

TCPS相对top-3的seed5/7增量为`+0.005705/-0.017417`，方向不一致。第三类间隔虽有非零权重，但不能稳定改善，IDEA-109拒绝并停止该上下文轴。

RUN-002模型SHA256：`645a589680e320603febca062ade012413447ca82da361e05bdc34c83423307b`；最后checkpoint SHA256：`d4b9b1fe995d6ef57389416e90ae66446e45db272adbaa1857ca6883bdf7b21f`。
