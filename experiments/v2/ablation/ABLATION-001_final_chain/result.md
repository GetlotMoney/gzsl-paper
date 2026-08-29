# V2-ABLATION-001 结果

```text
TG-VPR                       H=74.023182
+ TST                         H=76.984545  Delta=+2.961363
+ NTR                         H=77.086536  Delta=+0.101991
+ CCGR                        H=77.384331  Delta=+0.297795
total Delta H                              +3.361149
```

TST提供主要seen到unseen迁移增益；NTR提供较小但可重复的邻域路由修正；CCGR在五seed全部正增益，并在seed7进一步提高U/S/H/ZS。所有条件均使用official test参与选择，不能描述为blind-test消融。
