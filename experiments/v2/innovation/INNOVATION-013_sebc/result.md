# V2-INNOVATION-013 结果

状态：`rescue1_conservative_gamma_planned`。

RUN-001全局best保留关闭态，`U/S/H/ZS=73.985535/80.904585/77.290521/83.061785%`，selected_epoch=`0`、gamma=`0`。第一轮episode训练已把gamma推到`0.630104`并使H降到`75.987236`，后续均未恢复，故障诊断为偏置扣减量纲过大。

模型SHA256：`e52d483054aaa01f9444ee3dba9bc4fedd2f7ae2ada1eca7b09b9c9afa3d2c78`；最后checkpoint SHA256：`be53077b5fb7cdf7de956dcaad32a4b4c3563ae557873e2a322715748e365182`。

RESCUE-1只把max_gamma从2.0收紧到0.2；episode、输入、loss、seed、训练量和评估语义不变。
