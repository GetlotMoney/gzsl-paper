# V2-CONFIRM-007 结果

状态：`planned`。

训练1个完整150类TG父模型和3个仅见100类的fold TG父模型。fold父模型不使用official test选模；共享TST/NTR+CCGR在每折50个真正未见类上训练，最终只对完整推理模型按Chen-style official H保存全RUN一个best。
