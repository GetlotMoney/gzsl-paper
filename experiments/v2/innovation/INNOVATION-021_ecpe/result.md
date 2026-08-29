# V2-INNOVATION-021 结果

状态：`rejected_episode_opposite_direction`。

RUN-001第一轮episode训练即学到beta=`-7.385873`，随后接近`-10`；所有非零条件均低于关闭态，整次RUN best退回SEBC `U/S/H/ZS=75.772560/79.346550/77.518382/83.061785%`、beta=`0`，也低于CCPE `77.666533`。

class-exclusive fold父模型中的局部证据风险与主推理模型方向相反，该episode目标不能迁移，IDEA-055直接止损。模型SHA256：`29304cf8b5f048c149977031d24ffdf39ab169c6baff279b58ec50eedaa10d11`；最后checkpoint SHA256：`ccbaf7cce4193de85c7d246440989b25bf4c16c903493c5ce8e222948dbaf587`。
