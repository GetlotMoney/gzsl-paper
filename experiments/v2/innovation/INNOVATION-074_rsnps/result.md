# V2-INNOVATION-074 结果

状态：`rejected_below_top3_on_seed7`。

RUN-001（seed5）最高`U/S/H/ZS=76.876497/80.143785/78.476148/84.015107%`，selected iteration=`846`。相对稳定union top-3提高H `0.009438`，比union top-5最高低`0.004562`。

top-5图包含202条mutual边和596条one-way边；互惠加权后pair weight std由union top-5的`0.154466`降至`0.152381`。追加seed7决定是否替代top-3稳定结构。

模型SHA256：`31d7d98c7a23b5f1f2ad9b104843deab3ecce4a3002e5e2e1a1e81724760a5f7`；最后checkpoint SHA256：`0732a811a97694c19ce2193ae457c2396834ec4c73eb19b4a462eab5978f6502`。

RUN-002（seed7）最高`U/S/H/ZS=76.844305/80.079335/78.428474/83.949006%`，selected iteration=`1269`。相对C-RGWPS提高H `0.015765`，但比稳定union top-3低`0.017626`。

R-SNPS相对top-3的seed5/7增量为`+0.009438/-0.017626`，方向不一致，不能替代top-3。语义图家族首次union top-5加三次方法级补救已完成，固定top-3为稳定结构、top-5为最高seed观察，关闭图密度与权重轴。

RUN-002模型SHA256：`0150c11a99d9b1717ea82c78b8b875b6a8deeba85de2e42046b02e85b3201ddd`；最后checkpoint SHA256：`af9e939d5aa92f228eb8b12ca472075561cb11c8f1deab05652e6eab5c2ed35e`。
