# V2-INNOVATION-019 结果

状态：`stagewise_run004_planned`。

RUN-001在第一个评估点后因双beta字典仍使用单浮点日志格式而停止，未完成正式训练。按工程失败记为`failed_runtime`，不采信局部checkpoint、不计方法补救；修复日志格式后使用新RUN目录重跑同一条件。

RUN-002只修复日志序列化；模型公式、参数、输入SHA、seed、训练量和评估语义不变。

RUN-002联合训练得到`U/S/H/ZS=76.220822/78.957713/77.565132/83.165276%`，低于CCPE `77.666533`。best绝对beta=`9.984808/10`接近饱和，归一化beta仅`0.057909/2`，说明同一seen CE下强绝对分支吞掉弱归一化分支。

模型SHA256：`6b106721375889921845c470061ae29b3f45bb91937091d66f4d0960ece1fe57`；最后checkpoint SHA256：`3d0ff22d5b93127693211aeeda6e25d7c3916fa1638d3490c38bbd1477082a16`。下一补救固定CCPE已验证绝对beta，只训练归一化beta。

RESCUE-1固定CCPE RUN-003的绝对beta和模型SHA，只训练归一化beta；关闭新增分支时必须逐值复现CCPE `H=77.666533`。

RUN-003在参数更新前因`dspe.v2`未被旧形状检查识别为400维双路分数而停止，记为`failed_runtime`，不占方法补救。修复schema形状检查后用新RUN目录重跑同一分阶段条件。

RUN-004只修复双路score宽度契约；训练公式、父权重SHA、参数、seed和评估语义不变。
