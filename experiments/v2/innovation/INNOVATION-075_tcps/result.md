# V2-INNOVATION-075 结果

状态：`testing_seed7_reliability`。

首次启动在创建输出目录前因patch-free schema漏接线而`failed_pre_run`，没有开始训练且不计有效方法次数；commit `359bbc6`修复后使用同一未创建的RUN-001目录正式运行。

RUN-001（seed5）最高`U/S/H/ZS=76.949829/80.056471/78.472415/84.052283%`，selected iteration=`705`。相对稳定SNPS top-3提高H `0.005705`；第13维均值/标准差=`1.262952/1.144100`且selector权重=`0.042346`，特征有效但增益极小。

模型SHA256：`0680133380674c99b8b8beb768f56683e8458f700d87db6d369f78e5f3f428dc`；最后checkpoint SHA256：`8eabc875e9340c1d1ee7637741ebbbe63ba97c020c8b42f679cf41b81fb37d76`。
