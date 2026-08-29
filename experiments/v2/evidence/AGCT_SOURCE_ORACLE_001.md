# AGCT_SOURCE_ORACLE_001

parent: `SDCR seed5 / H=78.320510%`
gate: `AGCT train-error margin 25分位 / threshold=0.236863`
output_uri: `/data/lby/projects/cv_project/GZSL_Warehouse/diagnostics/v2/agct_source_oracle_20260824.json`
output_sha256: `bd017e6b25766e970eeedf6cf768a76127c85e3c789fdd772ffbd615f59913ab`

硬gate覆盖：seen `131/1764`，unseen GZSL `256/2967`，unseen ZSL `153/2967`。

关键事实：

- unseen GZSL gated样本中基线错误`140`个，top2包含真类`195`个；top2 oracle可净纠正`79`个并把理论H提高到`80.900744%`。
- Claude max/min净纠错分别为`-11/-26`，merge为`-15/-22`，patch为`-23/-14`；没有单一固定来源或方向能作为硬选择器。
- seen split只有Claude min得到`+4`净样本，但它在unseen为`-26`，不能作为跨域规则。
- ZSL patch min为`+3`，但GZSL unseen为`-14`，仍不稳定。

决策：保留AGCT软负beta辅助结果，但关闭固定来源硬tie-break。下一候选训练一个跨类别共享的小型pair selector，输入parent margin与三种source差值，直接学习top1/top2成对CE；真实unseen图像不进入梯度。
