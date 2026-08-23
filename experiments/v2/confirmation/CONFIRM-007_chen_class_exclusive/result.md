# V2-CONFIRM-007 结果

状态：`rescue1_high_transport_planned`。

训练1个完整150类TG父模型和3个仅见100类的fold TG父模型。fold父模型不使用official test选模；共享TST/NTR+CCGR在每折50个真正未见类上训练，最终只对完整推理模型按Chen-style official H保存全RUN一个best。

RUN-001结果：`U/S/H/ZS=73.860192/77.948201/75.849154/83.031738%`，未超过普通0.5分阶段父条件。RESCUE-1复用SHA绑定的完整TG和三个fold父模型，只把class-exclusive共享迁移步长上限从0.5恢复为1.5。
