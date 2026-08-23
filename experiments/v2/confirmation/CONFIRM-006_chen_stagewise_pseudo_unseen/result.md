# V2-CONFIRM-006 结果

状态：`completed_rejected`。

相对CONFIRM-005最佳RUN-002，唯一方法改动是在TRANSFER_CCGR阶段增加`0.25 × pseudo-unseen CE`。pseudo-unseen来自150个seen类的固定三折，真实unseen图像不进入梯度。

结果：`U/S/H/ZS=74.326867/77.642840/75.948676/82.930040%`，best位于iteration 8037/epoch 57/TRANSFER_CCGR。相对CONFIRM-005最佳0.5条件H下降`0.058172`，因此拒绝。

失败原因：同一个TG-VPR父模型已经见过全部150类，简单对batch中的pseudo-unseen样本加权没有形成真正class-exclusive迁移任务。下一Experiment必须让每折父模型只训练100类，另外50类对该父模型从未进入梯度。

模型SHA：`9c7da3580f4bd762433d33b51f2e95c6df1ae18d25f084a8f11430a253a2c42a`。
