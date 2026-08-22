# 三项核心创新claim边界

当前检索只能证明“已发现的最近先例与本项目的差异”，不能证明全球范围绝对首次。论文写作必须采用窄claim。

| 核心创新 | 可以主张 | 禁止主张 | 最近先例 |
|---|---|---|---|
| TG-VPR | 在GPT视觉描述基础上，固定划分local/unique/overall角色组，使用共享Value重参数化、等权组融合和topology约束构造GZSL原型 | 首次使用GPT描述、首次结构化prompt、首次多描述CLIP | PAPER-001、PAPER-004 |
| TST | 在inductive GZSL中，以Mean8为球面基点，只使用目标类Value切向分量；步长由seen内部三折pseudo-unseen episode训练，true-unseen图像不进入梯度 | 首次超球面prototype、首次测地线/切向插值、首次prototype transport | PAPER-005、PAPER-006 |
| CCGR | 对每个目标类别，只在其自身Value/local/unique/overall四个文本切向方向内生成有界残差，并以类别几何预测组合与幅度 | 首次动态语义prototype、首次双/多prototype、首次视觉增强prototype | PAPER-007、PAPER-008 |

辅助CRA/VPA/EBC/JBEC/CNRA只作为可靠工程组合和消融结果，不承担原创claim。

## 整体连贯表述

三项核心创新共同回答“如何把LLM视觉描述变成可迁移且受约束的GZSL类别原型”：TG-VPR建立角色化语义，TST在单位球面上安全迁移，CCGR在目标类自己的文本几何中做类别条件精修。辅助语义证据头只负责补充attributes、类名和联合竞争平衡。
