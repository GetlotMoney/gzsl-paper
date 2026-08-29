# V2-INNOVATION-072 结果

状态：`supported_top3_two_seed_patch_free`。

RUN-001（seed5）最高`U/S/H/ZS=76.916498/80.109864/78.480710/84.056246%`，selected iteration=`282`。相对SDCR父条件四项分别提高`0.169498/0.149971/0.160200/0.102270`，相对C-RGWPS提高H `0.087533`；它是当前绝对最高H，且不读取patch。

语义top-5对称化后得到798条无向边，关系扩展使有效pair从C-RGWPS的4041增至4691；门槛仅由473个seen训练错误关系样本的25分位得到。

模型SHA256：`37f60a715a700cf6e44a2e0b05a5b6b8b1c539d2aac6622f2cd0749c379cdd4b`；最后checkpoint SHA256：`0c20ba77b3bd9a0633f4655f326d03079b532a1e4772ae4b60543cb3a86911b3`。

RUN-002（seed7）最高`U/S/H/ZS=76.781482/80.059534/78.386251/83.953416%`，selected iteration=`282`；相对同seed SDCR提高H `0.083396`，但比同seed C-RGWPS低`0.026458`。

两seed绝对H范围=`0.094459`，按owner口径主成绩取seed5最高`78.480710%`。不过SNPS相对C-RGWPS的增量为seed5 `+0.087533`、seed7 `-0.026458`，方向不完全一致；因此保留为最高成绩候选，不作为稳定独立核心创新。

RUN-002模型SHA256：`c01ab7cebaa9513dad88d866ce0c77cca6bfac2efdc068043141cb5dda1ea9c0`；最后checkpoint SHA256：`977967ee3c6d14221954176bd563767cfbc67d5ac728a4aad85505ed95696cb3`。

RESCUE-2 / RUN-003（seed5）union top-3达到`U/S/H/ZS=76.883179/80.116844/78.466710/84.121209%`，selected iteration=`564`。479条语义边与4458个pair处于union top-5和mutual top-5之间；H也位于两者之间，但ZS为三种规则最高。追加seed7检验增量方向。

RUN-003模型SHA256：`08fa8b10b6285657c2f536c556b702d8256f7208dc1bfa61f3d90f628f5c81e9`；最后checkpoint SHA256：`c2f442da1d61c5c9c520c45f391dc57147e5da4f8a448c327aee38813fc72018`。

RUN-004（seed7）union top-3达到`U/S/H/ZS=76.847547/80.112571/78.446100/83.987880%`，selected iteration=`705`；相对同seed C-RGWPS提高H `0.033391`，并超过同seed mutual top-5 `0.019202`。

union top-3的seed5/7 H=`78.466710/78.446100`，范围仅`0.020610`，且两seed都高于C-RGWPS和mutual top-5；因此固定为SNPS家族的稳定结构。union top-5 seed5 `78.480710%`继续作为owner口径最高成绩，top-3作为更稳健的结构证据。

RUN-004模型SHA256：`1ef663533813e0995cd5a0b6ddb482b9e94be3287f4f26e058ac5e73f0359539`；最后checkpoint SHA256：`0515df724d6e8a6431187ff6a4bea42c62df0acc482d2e7b8c7f1a2e01083574`。
