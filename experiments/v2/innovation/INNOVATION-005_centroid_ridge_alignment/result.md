# V2-INNOVATION-005 结果

```text
highest H / seed        = 79.448210 / 17
H mean/min/max/range    = 79.377409 / 79.336822 / 79.448210 / 0.111388
U mean                  = 75.197838
S mean                  = 84.048975
ZS mean                 = 86.171220
Delta H min/max         = 1.786283 / 1.875527
```

四个seed的U/S/H/ZS均高于各自CCGR父条件。相较普通ARA，CRA降低属性融合强度并改善U/S平衡。类别中心ridge与属性映射均有传统方法先例，因此这里只声明当前框架中的可靠统计改进，不作核心算法原创claim。

Ridge收口：`lambda=0.01/0.1/1.0`的seed17 H为`79.448210/79.340503/77.699950%`，固定0.01。

输出契约：V2-TRY-111逐位复现seed17结果，并生成正式五件套。`ara_model.pth`与`model_best.pth` SHA均为`d7ac053e708037f1b43f1a8252ee9f94fa33cf302bc5bc6365ec92d77d841592`，`checkpoint_last.pth` SHA为`d5c6cc58d04724750ea2c977502b1d8687c3b64ee3437a367b88e90481fd4576`。
