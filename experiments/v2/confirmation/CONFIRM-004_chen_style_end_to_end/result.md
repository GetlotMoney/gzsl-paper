# V2-CONFIRM-004 结果

状态：`planned`。

两条RUN均端到端联合更新TG-VPR、TST/NTR和CCGR；专家条件额外更新312维属性残差。模块不分阶段选最大，只根据完整模型official H保存`model_best.pth`。

协议固定为Chen-style：trainval训练、每步独立随机抽50张、28,228次更新、每141步official test、`test_used_for_selection=true`。
