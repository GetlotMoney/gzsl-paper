# V2-INNOVATION-013 结果

状态：`supported_auxiliary_composition`。

RUN-001全局best保留关闭态，`U/S/H/ZS=73.985535/80.904585/77.290521/83.061785%`，selected_epoch=`0`、gamma=`0`。第一轮episode训练已把gamma推到`0.630104`并使H降到`75.987236`，后续均未恢复，故障诊断为偏置扣减量纲过大。

模型SHA256：`e52d483054aaa01f9444ee3dba9bc4fedd2f7ae2ada1eca7b09b9c9afa3d2c78`；最后checkpoint SHA256：`be53077b5fb7cdf7de956dcaad32a4b4c3563ae557873e2a322715748e365182`。

RESCUE-1只把max_gamma从2.0收紧到0.2；episode、输入、loss、seed、训练量和评估语义不变。

RUN-002得到`U/S/H/ZS=75.772560/79.346550/77.518382/83.061785%`，相对SDRS父模型U/S/H/ZS变化为`+1.787025/-1.558036/+0.227861/+0.000000`个百分点。最佳位于episode epoch=`1`，learned_gamma=`0.153261/0.2`，未贴边并通过预注册边界。

训练阶段只使用全局150个seen类图像，在三个100/50类class-exclusive episode中训练gamma；真实unseen图像不进入梯度。official test用于整次RUN的全局best选择，明确`test_used_for_selection=true`。

模型URI：`/data/lby/projects/cv_project/GZSL_Warehouse/innovation/v2/INNOVATION-013_sebc/RUN-002/model_best.pth`

模型SHA256：`63ff8397718b40d51b2698e2a6fa770cf78cff1913a9c99bc6920b06abe0309b`；最后checkpoint SHA256：`056df539a8d5051c5f2ef2e975d37571d63e45c4cda1b0de8998ae0cd997d51f`。

SEBC复用已有EBC公式，只证明其在当前无专家组合中有效，不作为首次校准或论文核心创新claim。
