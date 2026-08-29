# V2-INNOVATION-078 结果

状态：`retained_highest_seed_not_stable`。

RUN-003预注册：保持RUN-001的seed5父链与全部公式不变，只把selector训练随机种子改为6，用于最高值搜索和随机性审计。

RUN-003（selector seed6、seed5父链）最高`U/S/H/ZS=76.747000/80.020505/78.349575/83.953977%`，selected iteration=`282`。同一父链的selector seed5/6 H=`78.555039/78.349575`，差`0.205464`；结合seed7链`78.431289`，RDSS高峰不稳定。

按owner口径主成绩仍取seed5 `78.555039%`，但RDSS只保留最高seed观察，不作为稳定核心创新。RUN-003模型SHA256：`d14537ff386501d88026537a0757a139e8319fee4ddc41f14e388d43e720a929`；最后checkpoint SHA256：`73ff74a215be1e90e38fd22878f5e53c00293ec7e675672b3aa00dcd43589f6b`。

RUN-001（seed5）最高`U/S/H/ZS=76.915932/80.265528/78.555039/84.053975%`，selected iteration=`705`。相对稳定SNPS top-3提高H `0.088329`，相对原最高SNPS top-5提高`0.074329`，成为新的patch-free最高。

第13维raw role std均值/标准差=`0.010994/0.007064`，selector权重=`-0.061355`；模型学会在角色分歧更大时降低pair修正。追加seed7验证可靠性。

模型SHA256：`d8d0b22c719ba2b989b96794391266ca03b2fae031913677a3ba49518575b976`；最后checkpoint SHA256：`12524de60c1a0f79ef5ca47b1dbd5454c8526acca5b0a547a2327b3364d7caa0`。

RUN-002（seed7）最高`U/S/H/ZS=76.914233/80.009395/78.431289/84.020644%`，selected iteration=`705`；相对SDCR提高H `0.128433`，但比同seed稳定SNPS top-3低`0.014811`。

raw role std权重seed5/7均为负（`-0.061355/-0.046857`），机制方向一致；问题是联合重训旧12维权重扰动稳定父模型。按owner口径最高仍取seed5 `78.555039%`，下一独立Experiment采用分阶段：冻结SNPS top-3选择器，只训练新增尺度系数。

RUN-002模型SHA256：`51829eeacfdd68dc6126777beed05b7b066c7dda3d224412fc8982221a9162c5`；最后checkpoint SHA256：`828f6469d67b5cbaf6b3b10770a7ae79842ba74d29adc86ce575cfb31c0c36ce`。
