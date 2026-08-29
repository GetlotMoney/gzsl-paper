# V2-INNOVATION-010 结果

状态：`rejected`。

RUN-001 signed gamma结果H=`76.099469`，相对父模型提高`0.092621`，但best时gamma均值`-1.107964`且max_abs接近2，机制方向与饱和门槛均不成立。

RESCUE-1把gamma改为严格非负、上限0.5，零初始化仍严格返回父logits；禁止通过增强seen获得表面增益。

RUN-002结果H=`75.846458`，相对父模型下降`0.160390`。父模型并非简单seen偏高，强制压seen无效；SCCC轴关闭。
