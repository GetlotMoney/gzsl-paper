# V2-CONFIRM-008 结果

状态：`completed_below_target`。

所有超参数绑定`V2-TUNE-003/RUN-001`：topology=`0.1`、transport=`1.5`、CCGR幅度=`0.2`、epoch=`17`。模型从随机初始化使用完整150类、7057张seen图像训练；训练完成并写入checkpoint后，official test只运行一次。

该结果继续披露`historical_test_informed_architecture=true / strict_blind_claim=false`，不得用于反向修改本方法。

最终模型训练17轮，每轮完整且唯一遍历7057张seen图像；checkpoint完成后official test运行一次，得到`U/S/H/ZS=73.589277/76.080686/74.814246/80.644178%`。`test_used_for_selection=false / official_test_evaluations=1`。

该结果低于77%目标，也比旧`V2-CONFIRM-003`无专家validation-frozen H=`74.971312%`低`0.157066`。按冻结协议只记录，不据此修改当前超参数；后续新方法必须回到三折validation建立新Experiment。

模型SHA256：`3f6974688d349d8a294e509892c7b84dc8e8490bfa7ef7d6431ede98545984d6`；最后checkpoint SHA256相同。
