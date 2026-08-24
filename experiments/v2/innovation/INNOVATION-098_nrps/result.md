# V2-INNOVATION-098 结果

状态：`planned`。

错误审计：S-EDPS可处理top2错误中，seen仅61/155、unseen仅110/288获得正确负delta方向；全局放大delta会降低H，瓶颈是线性交互能力而不是幅度。

唯一改动：冻结S-EDPS的12维线性selector，在其raw score上增加`12→8→1`零初始化受限残差，最大raw残差`0.25`。只训练残差MLP；关闭时逐位复现S-EDPS。seed5须超过`78.572828%`才追加seed7。
