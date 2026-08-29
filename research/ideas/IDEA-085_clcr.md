# IDEA-085：Cross-LLM Complementary Residual

status: rejected
problem: SDCR使用GPT-5.6八句并取得最高H；OCLR使用独立Claude描述并明显提高U/ZS，但两者从未在当前父链上直接组合。
hypothesis: 固定SDCR后增加一条小幅Claude类名正交残差，可利用跨LLM互补信息，在保持S的同时提高U/ZS并超过SDCR。
evidence_refs: IDEA-063 OCLR两seed可靠提高U/ZS；IDEA-075 SDCR两seed可靠取得最高H；仓库检索未发现二者直接组合。
base_commit: d2ae02a53e9d66491863a46e7b546d4ca26711da
core_change: SDCR全部冻结；新增独立Claude正交原型，只训练范围±5的一个beta。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，beta不在±5边界。
failure_condition: H不超过SDCR、beta退回0/饱和或跨LLM分支只改变U/S权衡。
experiment: V2-INNOVATION-051
result: Claude与SDCR原型确实不同，但所有非零beta均降低H，best退回0；直接跨LLM叠加不互补。
