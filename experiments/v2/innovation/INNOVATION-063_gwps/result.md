# V2-INNOVATION-063 结果

状态：`testing_seed7_reliability`。

RUN-001将pair训练集从GPES的169扩展到`4041`，soft gate权重mean/std=`0.043862/0.160244`。最高`U/S/H/ZS=76.735932/80.086303/78.375328/84.009010%`，相对SDCR H提高`0.054818`，相对AGCT最高提高`0.018104`。

selector五参数有限，四维权重为`[-0.079356,0.007376,-0.425034,-0.163089]`。但pair中top1真类比例高达`0.931700`，需追加seed7验证是否偶然。

模型SHA256：`e0f359a8e78cb51aa4f3be909523fbb408b063d9440abaf3699d6622a6ddfaad`；最后checkpoint SHA256：`15c427f9c7c11dbf002d9b27e6ae9fcffc2d662e4de5741bba1afd763d6419ee`。

RUN-002使用seed7父链，pair数同为`4041`，最高`U/S/H/ZS=76.773667/80.126470/78.414246/83.980089%`，相对seed7 SDCR H提高`0.111390`，四项均提高。selector权重有限，selected iteration=`14664`。

两seed均提高H，GWPS标记supported辅助候选；当前最高可靠主成绩按owner规则取seed7 `H=78.414246%`。由于推理依赖来源不完整的patch cache，必须保持`feature_provenance_complete=false`，暂不作论文核心创新。

RUN-002模型SHA256：`a8245c2812bdd451b4967c113ee0997a4579a212b712d51aff1917889ca8e8e8`；最后checkpoint SHA256：`493f51c1a4ff904ee8b3ad995e1253b94f12a5a95daa5184960dec6c2cc82e56`。
