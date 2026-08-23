# V2-INNOVATION-038 结果

状态：`supported_seed7_reliability_seed5_pending`。

RUN-001得到`U/S/H/ZS=76.714796/79.713905/78.185600/83.888566%`。权重min=`0.111351`健康，但std仅`0.008806`低于0.01门槛，且H低于有效AOSR `78.210580`，说明KL=0.1使路由近等权。

模型SHA256：`3228d0846535d6ede23b9e626be384ca9f7611c489d001de16db5c38987408e7`；最后checkpoint SHA256：`241efbd3af07e645cf7fb6d9bc2ec95204b0c2b73595c6d3fcb82f7fe735a3b3`。RESCUE-1只把KL权重降到0.01。

RUN-002除kl_weight=`0.01`外，父模型、seed、数据、优化器和评估协议均与RUN-001相同。

RUN-002得到`U/S/H/ZS=76.849824/79.776293/78.285719/83.920640%`，超过全部既有有效结果。权重std/min/max=`0.034369/0.080386/0.184452`，同时满足差异和非塌缩门槛。

最佳位于iteration=`16215`。模型SHA256：`6056345e17786ee84e62d9489368ade4e1616b03b26f33d9d9741d77af6d2be5`；最后checkpoint SHA256：`4d6538c0ec4ef110039ee242bce9aa1f664bc730130c4d958c0a49a7e3a9ffc6`。追加seed5完整链可靠性后再正式晋级。
