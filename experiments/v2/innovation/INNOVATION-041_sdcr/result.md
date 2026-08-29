# V2-INNOVATION-041 结果

状态：`supported_two_seed_drop1_fixed`。

RUN-001得到`U/S/H/ZS=76.713103/79.959893/78.302856/83.920079%`，H比CASR高`0.017137`，属于弱增益。推理权重std/min/max=`0.055924/0.049531/0.233719`，未塌缩。

8句mask次数为`[3499,3481,3568,3487,3469,3614,3568,3542]`，训练dropout覆盖均衡。模型SHA256：`d1371389438b2f8b4b65f8735c683bd19ffe54f220ac593ee302d51c4d123773`；最后checkpoint SHA256：`2f7c074962bb2003f4138b9915661921029577e079876107fc36ab6bf9205392`。

因增益仅0.017137，追加seed5完整链可靠性后再决定。

RUN-002使用CASR seed5正式best与seed5随机链，其余dropout、KL和优化参数不变。

RUN-002得到`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`，超过seed5 CASR父模型。推理权重std/min/max=`0.055400/0.048246/0.229729`，mask计数`[3542,3463,3531,3609,3562,3540,3472,3509]`覆盖均衡。

seed5/7 H=`78.320510/78.302856`，差距`0.017655`且两条链都超过CASR，SDCR可靠成立。按owner规则正式最高取seed5 `H=78.320510`。

RUN-002模型SHA256：`53f9065ddd5f32bc02ff4be3ce5db3c7a4eadf5117282b55a672780acec001ae`；最后checkpoint SHA256：`8c61a107d2aedff8c4396b1706d4ba16f4fefedc510a67932416843bd9602aad`。seed7模型SHA为`d1371389438b2f8b4b65f8735c683bd19ffe54f220ac593ee302d51c4d123773`。

RUN-003只把每个训练batch的mask数量从1增到2，使用seed5 CASR父链；推理仍恢复完整8句。

RUN-003得到`U/S/H/ZS=76.713669/79.959893/78.303151/83.887309%`，高于CASR父模型但低于drop1 seed5 `78.320510`。权重std/min=`0.056328/0.042187`健康，mask计数总量约为drop1两倍且覆盖均衡。

dropout数量轴关闭，SDCR最终固定每批mask 1句。RUN-003模型SHA256：`db4244e23fea95bf2892976426d72da76d4a04e20aa8854776785bd97d00d82d`；最后checkpoint SHA256：`cfc052287d2e433c9b3d36761768928bc9dcd1633dbae7caa13938a0cfc7b116`。
