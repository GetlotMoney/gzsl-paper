# V2-INNOVATION-042 结果

状态：`rejected_consistency_overconstraint`。

RUN-001得到`U/S/H/ZS=76.679766/79.959893/78.285486/83.887309%`，仅略高于CASR父模型，明显低于SDCR `78.320510`。权重非塌缩、mask覆盖均衡，但显式教师一致性限制了dropout学生的有效偏移。

IDEA-076拒绝并关闭一致性loss轴。模型SHA256：`1c85287b06044436d45ed068669d184f8a182620447ec1cb28d4fbb254bd65bc`；最后checkpoint SHA256：`ffeaf33e3cec6dd41e8b6d12e1081f89c78159e568f8fc93a324a3a7870f6769`。
