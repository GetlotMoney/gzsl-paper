# V2-INNOVATION-050 结果

状态：`rejected_coordination_no_gain`。

RUN-001三个参数组均获得梯度，但所有联合更新条件都低于父模型，best严格退回`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`、selected iteration=`-1`。mask覆盖均衡，初始权重成功恢复。

训练过程中SEBC gamma从`0.153261`持续升到约`0.1853`且H同步下降，说明普通seen CE破坏了原本由class-exclusive episode学到的竞争偏置。RESCUE-1冻结SEBC，只协调SDRS一维斜率与SDCR八维句权重；其他条件不变。

模型SHA256：`607d73bf03270506f5e803a9ee5247714d739d9ba8f7abfdfdba744240522fdc`；最后checkpoint SHA256：`ae5f0f0897111d4de276a07b2afe8fb82adfc095151c132fa8860357183265e2`。

RUN-002冻结SEBC后，gamma稳定在`0.153261`，但所有更新条件仍降低H，best再次退回父模型。SDRS delta从`0.394185`持续降至约`0.30`，说明普通seen CE同样破坏已收口的类名缩放。最终RESCUE-2再冻结SDRS，只训练SDCR八维句权重。

RUN-002模型SHA256：`bde9af131bd7f26ecbe19cf600965c57011a0e14f7ffebc687647b2fa2bb42ec`；最后checkpoint SHA256：`bfe65bb4a37c789fc37db759541cde916f7ce8e75e93b225141707a45f909331`。

RUN-003进一步冻结SDRS，只以0.001学习率训练SDCR八维句权重；所有更新条件仍低于父模型，best第三次退回`H=78.320510%`、selected iteration=`-1`。这证明SDCR在现有dropout+KL目标下已收口，不是其他参数组冲突导致的假失败。

JSCF三个边界（10参数、冻结SEBC的9参数、只训SDCR的8参数）全部失败，IDEA-084拒绝并关闭分阶段协调轴。RUN-003模型SHA256：`32ba6585dd5c41b3aca9f5fbdf7fe469eb463cff91b9c01356672a376ef7fad1`；最后checkpoint SHA256：`6d96f17289bbee82a791936816a5751c607e3fecd090363d1acfdcd3194033aa`。
