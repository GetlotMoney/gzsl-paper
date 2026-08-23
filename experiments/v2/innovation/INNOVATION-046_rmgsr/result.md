# V2-INNOVATION-046 结果

状态：`rejected_family_budget_exhausted`。

RUN-001使用`coefficient_l2_weight=0.05`，best严格退回SDCR父模型`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`，selected iteration=`-1`，4个共享系数与class variation均为0。该正则消除了饱和，但同时压死了有效类别差异，判定为过强正则失败。

模型SHA256：`b64fc82dcb501a14df6debebc3345da0f985d81b6bb02f710d24cdea0fcddaae`；最后checkpoint SHA256：`75f583623e13ddab34848ec99beb26e208c79df4045ff79f4aa2b9cfa9d0cb80`。下一RUN是MGSR家族最后一次补救，只将L2降低10倍到0.005。

RUN-002使用`coefficient_l2_weight=0.005`，训练中产生过非饱和类别差异，但所有非零条件都未超过父模型；最终best再次严格退回`H=78.320510%`、selected iteration=`-1`、class variation=`0`。模型SHA256：`ec816464b614c375b64b6e4a02d2043ea45745a518c10389c2ed2859de5b1d9a`；最后checkpoint SHA256：`45e44e2dec6042893d10a60534fd77a76c2fdd30c2b6288b83d3ae01040753d2`。

MGSR家族结论：无系数正则时可观察到`H=78.365239%`但输出饱和；缩小输出上限仍饱和；强/弱系数正则均使best退回关闭态。3次补救预算已用完，IDEA-079/080均不晋级并强制止损。
