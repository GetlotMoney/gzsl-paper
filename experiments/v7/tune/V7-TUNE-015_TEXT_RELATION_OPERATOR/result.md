# V7-TUNE-015 Text Relation Operator

状态：`planned`。

本尝试只替换TUNE013的一文本关系头：删除逐图Reader路径，学习共享identity-residual低秩算子`D_vis = normalize(D_text + A(D_text))`，用trainval seen视觉中心差分监督seen-seen边。`A`零初始化时`D_vis`精确等于原一文本方向`D_text`，避免`normalize(0)`，再把同一个算子作用到全部seen/unseen相关边并经ridge编译为单个类别矩阵`Q`。

正式RUN尚未执行。本地最小测试只验证代码合同、梯度、导出等价和200类评估口径。
