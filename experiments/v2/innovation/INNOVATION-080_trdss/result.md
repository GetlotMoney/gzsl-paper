# V2-INNOVATION-080 结果

状态：`rejected_not_two_seed_consistent`。

RUN-001（seed5）最高`U/S/H/ZS=76.917648/80.219018/78.533653/84.054536%`，selected iteration=`282`。相对SNPS父模型提高H `0.066943`，比完全联合RDSS最高低`0.021386`。

role scale weight=`-0.037190`，旧12维weight drift=`0.115727`、bias drift=`0.007540`，信赖域允许有限协调。追加seed7检验稳定性。

模型SHA256：`35d914c74798775814ea657c86084d2dcb06d8df20cac28cb7e860b38621e96f`；最后checkpoint SHA256：`4b133f112620edc1ce77e5e7481c7834ebbe1d6fcb33328b2f6399399f7db01d`。

RUN-002（seed7）完整训练后best严格为SNPS父模型`U/S/H/ZS=76.847547/80.112571/78.446100/83.987880%`，selected iteration=`-1`；role scale weight、旧权重drift和bias drift均为0。

TR-RDSS相对父模型的seed5/7增量为`+0.066943/+0.000000`，未满足两个seed均正。信赖域只改善seed5，IDEA-114拒绝，不替代SNPS稳定结构。

RUN-002模型SHA256：`a4aa50791597670d011d9de1e5b62b0ab8f1f389ee9924b7adcf251c4168ffac`；最后checkpoint SHA256：`c5aa4452218500aa2dbb57061c161438d54d2001b465a26b451b69c88462dcea`。
