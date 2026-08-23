# V2-INNOVATION-057 结果

状态：`rejected_pairwise_graph_harm`。

成对affinity平均熵=`0.731696`，确认不是均匀矩阵；alpha由seen CE推到约`0.22`，但H长期约`76.9%`。所有非零条件都低于父模型，best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`、selected iteration=`-1`、alpha=`0`。

TWLS与TPLD证明：无论统一还是语义加权，固定族群图对所有样本做高通都会放大错误邻接。IDEA-091拒绝并关闭固定图结构轴；下一方法只能针对低margin样本动态启用独立证据。

模型SHA256：`fa5c83ed8244663f19f623aefff2888f3da0ca765252d24ccccd2b296225f650`；最后checkpoint SHA256：`17a80f0b7af26077e54cdd9a346eda79798ea6705e269f9918eccae7cef8f7b8`。
