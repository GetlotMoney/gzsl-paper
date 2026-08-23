# V2-INNOVATION-048 结果

状态：`rejected_wrong_placement`。

RUN-001所有非单位度量条件都低于SDCR，best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`，selected iteration=`-1`，最终权重全1。训练期间weight std稳定约0.009，证明参数真实更新；失败来自只变换SDCR残差分支，破坏其与TG主原型、SDRS类名残差和SEBC偏置的既有尺度平衡。

IDEA-082拒绝，不追加相同位置参数补救。下一实验若继续对称度量，必须把同一度量同时施加到完整三个原型分支，并保持seen偏置不变。

模型SHA256：`d8d3d321c3575581107788f446e724625ceb7cb4e20929dd53e0b19ad3514dd3`；最后checkpoint SHA256：`a9aeb70aa11fdefc124336be632609e5ed359059da9181d93890e6ad87247907`。
