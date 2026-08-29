# V2-INNOVATION-023 结果

状态：`rejected_seen_visual_domain_bias`。

RUN-001中beta被seen CE持续推到`10`上限，但所有非零条件均降低H，最终约`76.946222%`；整次RUN best退回SEBC关闭态`U/S/H/ZS=75.772560/79.346550/77.518382/83.061785%`、beta=`0`。

seen局部视觉中心ridge生成unseen原型仍产生与全局SVPG相同的域偏置，IDEA-057拒绝且不调ridge。模型SHA256：`64e75b84f234d18b99692c304fe162536977e5af50f51a8896817f13dcdcb99f`；最后checkpoint SHA256：`5c7baad66402c048d11a8be656529371882955edcc7df2c5f492029e7121655e`。
