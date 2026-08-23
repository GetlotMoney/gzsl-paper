# V2-INNOVATION-046 结果

状态：`testing_final_rescue`。

RUN-001使用`coefficient_l2_weight=0.05`，best严格退回SDCR父模型`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`，selected iteration=`-1`，4个共享系数与class variation均为0。该正则消除了饱和，但同时压死了有效类别差异，判定为过强正则失败。

模型SHA256：`b64fc82dcb501a14df6debebc3345da0f985d81b6bb02f710d24cdea0fcddaae`；最后checkpoint SHA256：`75f583623e13ddab34848ec99beb26e208c79df4045ff79f4aa2b9cfa9d0cb80`。下一RUN是MGSR家族最后一次补救，只将L2降低10倍到0.005。
