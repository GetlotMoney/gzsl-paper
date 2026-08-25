# V2-CONFIRM-012 结果

状态：`planned`。

本实验只重新计算三数据集Pure CLIP B0。图像缓存、标签和八角色文本embedding与`V2-CONFIRM-010`逐文件复用；唯一变量是使用冻结text-v2共同前缀重新编码的`class_name_embeds.pt`。B1与M3不重训，完成后只重新计算其相对公平B0的差值。
