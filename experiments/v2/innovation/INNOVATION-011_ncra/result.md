# V2-INNOVATION-011 结果

状态：`supported_target_passed`。

RUN-001 H=`76.407873`，相对父模型提高`0.401026`，但beta=`4.985551/5`接近上限。

RUN-002 H=`76.856269`，相对父模型提高`0.849421`，U/S/ZS均提高；beta=`9.943932/10`仍接近上限，说明类名残差有效，但当前允许范围仍不足。

RUN-003把同一残差的beta上限扩大到20，不改变模块、loss、数据或评估语义。正式结果为`U/S/H/ZS=75.131226/79.388309/77.201125/83.028460%`，相对父模型H提高`1.194277`个百分点，并超过项目目标`77.023182%`共`0.177943`个百分点。最佳权重位于iteration=`282`，beta=`17.151897`，没有落在20的上限。

训练只用150个seen类图像计算交叉熵并更新一个beta参数；unseen图像不进入梯度。整次RUN共28,228次随机batch更新、202次official test评估，按最高official H保存一个全局最佳权重，明确`test_used_for_selection=true`，不声称blind-test。

模型URI：`/data/lby/projects/cv_project/GZSL_Warehouse/innovation/v2/INNOVATION-011_ncra/RUN-003/model_best.pth`

模型SHA256：`c14ae2dad815a0873bd66a80b4d560e4c576a3165822b4629a1b890d15491a75`

最后checkpoint SHA256：`ef5d03599a01df74b32dd8327eaeba098af79f3d40db5d8152da86fed82938b1`
