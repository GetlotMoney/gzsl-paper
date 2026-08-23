# V2-CONFIRM-005 结果

状态：`planned`。

固定50/100/50名义epoch三阶段；阶段边界不根据test移动，阶段间使用最后权重继续训练。全程只有一个跨阶段的整模型best-H，不为每阶段分别保存test最大父checkpoint。
