# V2-ABLATION-002 结果

最终最高seed链从TG-VPR基线`H=74.023182%`提高到`80.712565%`，总增益`6.689383`个百分点。

```text
TG-VPR → TST                 +2.961363
TST → NTR                    +0.101991
NTR → tuned CCGR             +0.486146
CCGR → CRA                   +1.875528
CRA → VPA                    +0.095399
VPA → VEBC                   +0.930471
VEBC → JBEC                  +0.008688
JBEC → CNRA                  +0.229797
```

论文核心创新仍只有TG-VPR、TST、CCGR。CRA/VPA/JBEC/CNRA统一称为辅助语义证据头：双向attributes构建补充证据，episodic标量训练平衡联合竞争，class-name语义补充类别身份。辅助方法已有先例，不作新的核心claim。
