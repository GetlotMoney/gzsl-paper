# FRAMEWORK-V1 模块

完整 HTML 结构图：[`framework_diagram.html`](framework_diagram.html)。

| Module | 中文作用 | 关闭时行为 |
|---|---|---|
| Frozen CLIP feature input | 提供冻结的 CLS 与 576 个 patch 特征 | V1 不训练 CLIP backbone |
| PSE | 用多句文本增强 seen 类原型 | 返回原始文本原型路径 |
| ICSA | 根据图像 CLS 调整类别语义 | 返回 PSE 后共享原型 |
| FGVD | 选择并编码局部 patch | 不形成几何局部记忆 |
| BVSA | 双向对齐视觉局部与文本语义 | 不产生局部分数 |
| SGMP | 提供局部语义辅助训练目标 | 对应辅助 loss 为 0 |
| Global/local fusion | `global + 0.2 * local` | local 权重为 0 时只剩全局分数 |

## 已迁入但尚未接入V1的独立模块

| Module | 身份 | 当前状态 | 入口 |
|---|---|---|---|
| TG-VPR-H1 | 创新模块1 | 独立代码与配置已迁入；未替换V1的PSE，也未分配Innovation编号 | `docs/TG_VPR_H1.md` |
