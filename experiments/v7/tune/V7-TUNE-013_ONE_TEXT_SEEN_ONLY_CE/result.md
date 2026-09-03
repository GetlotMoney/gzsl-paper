# V7-TUNE-013 结果

状态：completed / drop_below_formal_v7_floor。

唯一改动：相对TUNE012一文本Full，最终head分类CE仅在150个trainval seen logits上计算；关系方向CE、统一八角色关系、训练预算和200类official推理保持不变。正式RUN使用冻结commit `35cefc52896c383e1ec75a3adc5f78d218d616a3`、config SHA `cadceb65cb9596ea8f9769424154636938120691f306766c8c769168d1f711b3`，从seed7原始初始化训练28,228 updates并完成201次official评估。

## 最佳结果

- best update/epoch：`13818 / 98`
- U/S/H/ZS：`79.960012 / 79.931587 / 79.945797 / 87.821192`
- 相对TUNE012 all-class Full `77.405741`：`+2.540056 H`
- 相对TG+GTD+S retrained `79.768159`：`+0.177638 H`
- 相对formal V7 `80.510432`：`-0.564635 H`

同checkpoint推理诊断：S/V/I-off H分别为`79.260523 / 78.941824 / 78.751928`，对应Full-minus-off为`+0.685274 / +1.003974 / +1.193869 H`；signflip与role shuffle分别比Full低`5.547975 / 2.002003 H`。这些关闭值只作为机制诊断，不替代从头重训消融。

## 判断

seen-only head CE基本消除了TUNE012的seen偏置：最佳U/S仅差`0.028425`，并让V/I即时贡献由负转正。这证明训练时把50个true-unseen logits持续当负类，是原一文本Full失败的主要原因之一。

但H仍低于formal V7 `0.564635`，预注册准确率地板未通过，因此程序decision=`drop_tune013_contract_failed`，不得接纳为正式框架或继续在该失败代码上堆叠。剩余差距可能来自uniform八角色关系方向相对专用比较句的信息损失，或Reader方向CE的跨类迁移不足；本实验不能在两者之间作因果区分。

## 产物

- output：`/data/lby/projects/cv_project/GZSL_Warehouse/tune/v7/V7-TUNE-013_ONE_TEXT_SEEN_ONLY_CE/V7-TUNE-013-CUB-ONE-TEXT-SEEN-CE`
- metrics SHA：`08fd38f284eced2ccd4ea49e1faae1f1163ce5b3821eb7373f9acedf9201f34e`
- model SHA：`7349e869c1ffef41c3cfe265486880a1c455feb442ff326ae8403bd9a9a39328`
- history SHA：`c3326575a5807ecb9ed5251b7012b62db2aac779a3f6d8d72af57bcc583e83fd`
- pre-review无效9024步输出保留于同级`.pre_review_invalid_20260903`，未混入正式结果。
