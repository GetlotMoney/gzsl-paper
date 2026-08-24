# V2-TUNE-002 结果

状态：`planned`。

外层固定使用xlsa17标准100/50类别不相交validation。内层三折只覆盖外层100个训练类，pseudo-unseen类别数为`34/33/33`，外层50个validation类不进入任何梯度。

RUN-001唯一新增训练语义：每个主训练batch额外计算三个inner fold的平衡pseudo-seen/pseudo-unseen CE，三折取mean并以固定权重`1.0`加入总loss。official test不会被加载。

对照条件为`V2-TUNE-001/RUN-001`：`U/S/H/ZS_val=76.424742/76.521248/76.472964/79.934835%`。
