# V2-INNOVATION-044 结果

状态：`rejected_below_sdcr`。

RUN-001最高为`U/S/H/ZS=76.713103/79.959893/78.302856/83.953977%`，best位于iteration `1974`。相对CASR seed5父模型H提高`0.026160`个百分点，但低于SDCR seed5最高`78.320510%`，未通过预注册门槛。

最终八句权重`std/min/max=0.057511/0.044432/0.236177`，mask次数为`[1228,6995,3982,2977,2262,1839,4572,4373]`。采样次数与句权重明显相关且每句均被训练，证明“重要句优先mask”真实生效；但它没有优于均匀随机SDCR，因此IDEA-078拒绝并关闭mask采样分布轴。

该RUN只使用seen训练图像，unseen图像未进入梯度；official test评估`202`次并用于选择iteration，明确`test_used_for_selection=true / strict_blind_claim=false`。首次以文件路径调用时因模块搜索路径失败，未创建输出目录、未发生训练，记为`failed_pre_run`且不计方法尝试。模型SHA256：`672ba9dc09b19ef23280847cee19a227f5a56e6336765b61f2d56b23c348717a`；最后checkpoint SHA256：`bd47d044f6570f823d660a2d75b0dfdacc0eaae2e65c80b3c7b8057fd95c31a2`。
