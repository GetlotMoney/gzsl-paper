# FRAMEWORK-V2 模块

完整结构见 [`framework_diagram.html`](framework_diagram.html)，可编辑源为 [`../../docs/TG_VPR_H1_framework_diagram.drawio`](../../docs/TG_VPR_H1_framework_diagram.drawio)。

| Module | 作用 | 输入 | 输出 | Baseline-off |
|---|---|---|---|---|
| Frozen CLIP CLS | 提供冻结图像表征 | `[B, 768]` | 归一化图像向量 | V2不训练视觉backbone |
| Eight-role semantics | 提供八种鸟类语义角色 | `[200, 8, 768]` | 八句类别语义 | 缺少固定顺序时拒绝运行 |
| Tri-group grouping | 把语义组织为局部、独特、整体三组 | 八句语义 | `[200, 3, 768]` | 回到Mean8原型 |
| Shared Value path | 对三组语义做共享768维Value重参数化 | 三组语义 | 三组上下文残差 | 关闭后不产生重参数化残差 |
| Fixed equal fusion | 三组固定各占`1/3` | 三组增强语义 | seen类融合语义 | 回到三组基准均值 |
| Seen prototype adapter | 用`0.35/0.65`外部残差形成seen原型 | 基准与增强语义 | seen类原型 | seen类回到基准原型 |
| Mean8 unseen fallback | 保持unseen语义不受训练改写 | unseen八句均值 | unseen类原型 | 本身就是baseline路径 |
| Topology loss | 约束200类关系结构 | 改写前后原型 | `0.1 × L_topology` | 权重为0时只剩CE |
| Cosine classifier | 计算图像与200类原型相似度 | CLS与类别原型 | train `[B,150]` / eval `[B,200]` | 无替代分类器 |

