# V2-INNOVATION-099 结果

状态：`planned`。

审计显示“文本距离最大角色”的方向在unseen可达错误上正确125/288，其中71个是S-EDPS方向错而该角色方向对；同时有476个正确样本可由它修复S-EDPS风险，具备互补性。

唯一改动：冻结S-EDPS 12维selector，只新增并训练一个top-discriminative role差值系数。关闭时逐位复现S-EDPS；seed5须超过`78.572828%`才追加seed7。
