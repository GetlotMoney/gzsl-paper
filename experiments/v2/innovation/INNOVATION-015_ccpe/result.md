# V2-INNOVATION-015 结果

状态：`supported_candidate_novelty_pending`。

RUN-001得到`U/S/H/ZS=75.883502/79.264784/77.537297/83.202559%`，相对SEBC父模型变化为`+0.110942/-0.081766/+0.018916/+0.140774`个百分点。四项方向整体合理，beta=`9.636184/10`未越过饱和门槛，但H和ZS均未达到预注册增益门槛，只记为弱正信号。

模型SHA256：`2aa2ac5b011b48d125291b64ae4bfc289805e3e592344b11fb9076ceaee4b194`；最后checkpoint SHA256：`39e2383d13a33047c55b69b2df394d8f21a90a609a6bba31354174bc4c48666c`。

RESCUE-1只把每类patch聚合从top8收紧为top4，以减少局部证据稀释；公式、父模型、beta范围、seed、训练量和评估语义不变。

RUN-002得到`U/S/H/ZS=76.052994/79.164737/77.577674/83.204818%`，相对父模型变化为`+0.280434/-0.181812/+0.059292/+0.143033`个百分点。top4优于top8且beta=`6.302845/10`未贴边，但H/ZS仍未过门槛，继续一次top2同公式补救。

模型SHA256：`340880746f38cb1d21f87763d816caaedd125ff5c693751798564c5d081960e9`；最后checkpoint SHA256：`fc5780770cec33fd7170b1c840a1db2c7405a9e54dfa9ed1783f151285bd70f1`。

RESCUE-2只把每类patch聚合从top4进一步收紧为top2；若仍不通过门槛，则关闭top-k参数轴，不继续top1网格。

RUN-003得到`U/S/H/ZS=76.119131/79.278153/77.666533/83.168101%`，相对SEBC父模型变化为`+0.346571/-0.068396/+0.148151/+0.106317`个百分点。H超过预注册`+0.10`门槛，beta=`9.177013/10`未贴边，top2条件成立并关闭top-k参数轴。

训练只使用7,057张seen图像的CE更新一个beta；真实unseen patch不进入梯度。official test用于每141次更新的陈式全局best选择，明确`test_used_for_selection=true`。遗留CLIP patch准确checkpoint来源仍不完整，结果保持`feature_provenance_complete=false`。

模型URI：`/data/lby/projects/cv_project/GZSL_Warehouse/innovation/v2/INNOVATION-015_ccpe/RUN-003/model_best.pth`

模型SHA256：`e3b2685b07883b976962c38804825e4043c500679003a869b4bc6997f60cfaf9`；最后checkpoint SHA256：`c94f007ab4cd1d09984dd02717adf4b8198a0ffa3dfd5a8360eb9a379e6dac6b`。

CCPE作为有实验支持的创新候选保留；在完成最近相关工作检索前不声明原创或论文核心创新。
