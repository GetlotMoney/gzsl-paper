# V2-INNOVATION-089 结果

状态：`rejected_below_top3`。

RUN-001最高`U/S/H/ZS=76.814806/80.172402/78.457698/84.121209%`，selected iteration=`564`，比稳定SNPS top-3低H `0.009012`。

bias固定为0且只有12维weight可训练，工程边界有效；但移除bias未改善H，说明bias差异不是跨seed不稳定主因。IDEA-123拒绝，不追加seed7。

模型SHA256：`3f8d452cbee1ca81a1685e4d093b14b07878147e19371a4dc3c635867fc5acd9`；最后checkpoint SHA256：`dbd72b9ee5a49085f16c90083420bc7ca0fad4bd15af41298444ecdc7505aa9a`。
