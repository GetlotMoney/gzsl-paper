# V2-INNOVATION-053 结果

状态：`rejected_uncentered_loss_scale`。

RUN-001的patch可靠性权重具有真实差异：mean/std/min/max=`1.164163/0.110662/0.758039/1.249999`。但所有训练条件都低于父模型，best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`、selected iteration=`-1`。

故障是权重均值高于1，除了样本相对加权，还把CE整体放大约16.4%，相对削弱固定KL。由于修复会改变loss语义，IDEA-087本实验拒绝；另建中心化权重Experiment，要求均值严格为1且边界仍受控。

模型SHA256：`f58fe4ce8a8e4fac0fd8f3f5ff0da9f3be655b53b9c8b5995ca9ec5122a4b966`；最后checkpoint SHA256：`ce725a744cc548f826c3872439cebeee7c700d1fc51b4b340f3bb60c82daadba`。
