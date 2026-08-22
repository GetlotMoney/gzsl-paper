# IDEA-038：JBEC联合双向校准

```yaml
idea_id: IDEA-038
source_type: sequential_auxiliary_optimization
evidence_refs: [V2-INNOVATION-007, V2-TRY-121]
base_commit: 2a34f51563bf6e827ea732d6cf0e25ba232bc7e9
problem: VPA beta先用seen CE训练、EBC gamma后用episode补偿，串行目标可能不是联合最优。
hypothesis: 在VEBC最优点附近用pseudo-unseen episode联合微调beta与gamma，可协调类内判别强度和域间平衡并进一步提高H。
core_change: 冻结所有特征、原型和正反ridge，只训练beta范围+-2与gamma范围+-0.05的两个零初始化残差。
success_condition: seed17 H超过80.474080%，U/S任一下降不超过2个百分点，两个残差均不饱和。
failure_condition: 首次TRY和最多3次方法级补救后仍不超过VEBC父条件。
status: testing
paper_core_innovation: false
parent_condition: V2-TRY-121 / TG-VPR + TST + NTR + CCGR + CRA + VPA + EBC
current_attempt: none
last_attempt: V2-TRY-131
last_decision: run_reliability_seeds
```

JBEC的正反ridge只用pseudo-seen中心重建，两个残差的梯度只来自seen图像；true-unseen图像不进入训练。该实验只检验辅助模块联合优化。

## V2-TRY-131结果

第20轮得到`U=77.045119%`、`S=84.241509%`、`H=80.482768%`、`ZS=87.227464%`，相对VEBC仅提高H `0.008688`；beta/gamma残差=`1.562304/0.047924`，均接近各自98%门槛。按预注册条件通过但效应极小，必须完成其余seed后再判断是否保留。
