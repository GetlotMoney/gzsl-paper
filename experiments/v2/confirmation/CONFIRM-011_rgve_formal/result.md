# V2-CONFIRM-011 结果

状态：`partial_local_hardware_observation`。

本实验只回填已经完成的正式RUN。RGVE-off/on必须使用同一新patch资产、同一代码语义commit、同一seed和同一训练策略比较。

## 本地RUN-001观察

本地RTX 5070 Ti完成了端到端RGVE-off：`U=73.431325 / S=80.033273 / H=76.590293 / ZS=84.486163`，best epoch为98。该RUN使用提交`3ffb59248d397d82879e0d2bc43bbc4a645ee814`，完整201次official评估，best-H对应指标来自同一checkpoint；独立best-ZS为87.066782，不与主指标拼接。

该结果只保留为`local_hardware_observation`。未来服务器4090正式2×2必须全部重跑，禁止将本地RUN-001与服务器RUN拼表。
