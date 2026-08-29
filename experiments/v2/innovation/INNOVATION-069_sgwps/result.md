# V2-INNOVATION-069 结果

状态：`supported_two_seed_patch_free`。

RUN-001全程不读取patch，使用4041个soft-gate pair。最高`U/S/H/ZS=76.747000/80.059719/78.368367/83.953977%`，比patch-free AGCT提高H `0.011143`，比SDCR提高`0.047857`。

RUN-002（seed7）最高`U/S/H/ZS=76.713103/80.059719/78.350691/83.920079%`，selected iteration=`282`，比同seed SDCR父条件提高H `0.047835`，比同seed patch-free AGCT提高H `0.011168`。seed5与seed7均为正提升，S-GWPS作为patch-free辅助候选成立；两seed最高取seed5 `H=78.368367%`。由于增益较小，暂不作为论文核心创新。

RUN-001模型SHA256：`67917f3b2915e9dbcec6b43f78a1ba5f513d30c75f71ccdbc90766ed1979b795`；最后checkpoint SHA256：`06da2991ab62a5ce1d42dd4f796b540bb4e06d7c8dc710980928140e8f1f34a3`。

RUN-002模型SHA256：`5925abff7a47ffeea86905d1c29f8b5f6ee9abc8ea771b63918178af787a728d`；最后checkpoint SHA256：`c5192437478726d9f0a7bf321853af7785b2ff7e340256a8fd030b5ca806d81e`。
