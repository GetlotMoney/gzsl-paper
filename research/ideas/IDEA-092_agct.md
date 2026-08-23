# IDEA-092：Ambiguity-Gated Cross-LLM Tie-Breaker

status: supported
paper_core_innovation: false
problem: 固定族群变换会伤害大量高置信度正确样本；错误审计显示真正需要处理的是top2同族且低margin的少数歧义样本。
hypothesis: 只在同族低margin样本的top2内调用独立Claude正交分数做二选一，可改变错误排名，同时保持其他样本和top2均值不变，从而超过SDCR。
evidence_refs: SDCR_ERROR_AUDIT_001；IDEA-085证明Claude全局叠加无效；IDEA-090/091证明全样本族群变换无效。
base_commit: 45394ae01e29a929d4179d469b7bbe3abede07c6
core_change: SDCR冻结；门槛仅由seen训练margin固定，Claude校正只作用于同族低margin top2，训练一个范围±5的beta。
success_condition: H大于78.320510，U和S任一项下降不超过2个百分点，official gate均值大于0且beta不在边界。
failure_condition: H不超过SDCR、beta退回0/饱和、gate近零或Claude在歧义样本仍给出错误排序。
experiment: V2-INNOVATION-058
interim_result: seed5 H=78.339523，相对SDCR +0.019013；gate非零且beta不饱和，但增益弱，追加seed7。
result: seed7 H=78.339523、相对父模型+0.036667；两seed均正且最高一致，supported辅助候选。
