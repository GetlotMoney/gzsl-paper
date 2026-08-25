# V2-CONFIRM-012 结果

状态：`running_after_failed_runtime`。

本实验只重新计算三数据集Pure CLIP B0。图像缓存、标签和八角色文本embedding与`V2-CONFIRM-010`逐文件复用；唯一变量是使用冻结text-v2共同前缀重新编码的`class_name_embeds.pt`。B1与M3不重训，完成后只重新计算其相对公平B0的差值。

首次启动的CUB与AWA2配置使用了运行器未注册的condition ID，实际回退为Mean8原型；两项产物保留并标记`failed_runtime`，不作为B0结果。修正后的运行器会在此类歧义发生时直接拒绝配置，正式重跑使用全新输出目录。
