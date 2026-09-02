# IDEA-221 / Semantic Candidate Local Verification (SCLV)

- status: `rejected_after_full_gate`
- problem_category: `reliability_robustness`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-220 / RG-DCF`
- performance_status: `below_parent`

## 路径

SCLV把全空间S/V融合改成生成—验证：冻结mean8语义S先产生Top-K候选，shared-coordinate局部视觉V只能在候选内选挑战者，OOF pairwise reliability I只决定保留S top1或换成该挑战者。

- old_solution_path: 在200类全空间判断是否整体信任弱V。
- new_solution_path: S生成小候选集 -> V局部验证 -> I执行一次有界换位。
- principle_difference: V从全局分类器变为受S搜索空间约束的验证器。
- non_equivalence_test: Full必须胜S、prior RG-DCF、同覆盖随机候选、语义/V-gap规则、V-always、打乱门控和打乱V特征。
- minimal_viability: Gate0候选可达空间与Full三模块移除共同过门。
- current_advantage: 候选空间极强，Full有小幅正增益，但不足1H且视觉归因不稳健。
- failure_boundary: 当前V仍使用绝对类别相似度，未直接学习细粒度候选对之间的区分证据。
- why_not_module: Full未达到模块+1门，不能登记为创新。
- paper_level_claim: none。

## 审核与代码身份

- 最终Idea草稿SHA：`8c7863baf6eb64487af998ea5cd09c23a90647a27a8d21f5664321c8dd4ad805`；A/B最终审核SHA：`09f82b8315c22d1208d58e3cdbd106834ca1bd10fcd32d6817277fbae308544c` / `2a8d6c916b09020b5127acb1105bb3eefbc45bfc368c1646c14045fc477c661d`，均P0/P1/P2=0。
- Gate0脚本SHA：`be202d4fe1d5af45f1d5c97e50ddeb8f6432ec6c5d70afb6577dcde9ff3a47bc`；交叉代码审查P0/P1=0。
- Full脚本SHA：`3821153a981db2e383199e737728fd3eafa1a3b1597d851d432caf880c49b6ca`；最终A/B代码审查SHA：`c899156df7d07701bb1341b6edbeb4f93edceebc008f350980338483649539d8` / `0b8be1a3495be7a179f581b593ff13f9a0b6f4e6fec6d03a1015f81498bfea49`，均P0/P1/P2=0。

## Gate0：语义候选可达性

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/sclv/IDEA-221-GATE0/result.json@sha256:1200cc7bf25ba31abb0b0bdaa6fffa698b9fd2cab0dfd194872bf3cc98d8ee54`。

- K3：seen/unseen中S错真类可达率`66.44%/58.42%`，TopK oracle `H=88.2128`，相对S `+19.4622H`。
- K5：可达率`84.34%/78.55%`，oracle `H=94.1794`。
- K2：oracle已达`H=81.3284`，但split-wise条件可达率`41.997%/38.834%`，未过预注册45%门。
- K3/K5/K8/K10通过；K20只作诊断。Gate0证明语义候选不是瓶颈。

## Full结果

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/sclv/IDEA-221-FULL-GATE/result.json@sha256:db9089c7c368adc1a5e63adb34a60f142713c2a5986d1cf3c379afede9b6fd9f`；日志SHA：`bbc18dc94a35ded2baf31c9ebd14a774bd30c3da162c4dbf0d56537d88dcacb0`。

| 条件 | U | S | H | ZS |
|---|---:|---:|---:|---:|
| class-name | 62.2104 | 64.2058 | 63.1923 | 79.6813 |
| mean8 S | 69.1363 | 68.3691 | 68.7506 | 86.1468 |
| prior RG-DCF | 69.1696 | 68.3691 | 68.7670 | 86.1468 |
| Full SCLV | 69.4052 | 68.4297 | 68.9140 | 86.1262 |
| TopK V-always | 35.9637 | 54.8089 | 43.4301 | 59.4759 |
| TopK oracle(K10) | 97.9704 | 98.4299 | 98.1996 | 99.7983 |

OOF选择`K=10,q=0.9`，恢复36/损坏22，coverage1.105%；pair AUC随K从K2的0.792上升到K10的0.876。official seen恢复9/损坏9，unseen恢复14/损坏6；Full相对S仅`+0.1634H`，bootstrap下界为负，未过1点门。

Full显著胜shuffled gate约`+2.61H`，且胜同覆盖随机候选约`+0.46H`、bootstrap下界为正，说明候选换位与可靠性并非随机；但`V_feature_shuffle H=68.7577`与Full差异不显著，说明绝对V分数尚未提供足够稳定的细粒度视觉区分。

下一补救保留已证明的Top-K生成路径，将V训练对象从“单类别绝对分数”改成“incumbent与challenger之间的反对称局部视觉似然比”。

披露：`test_used_for_selection=true`、`test_used_for_hyperparameter_selection=true`、`nested_official_test_selection=true`、`unseen_images_used_for_gradient=false`、`strict_blind_claim=false`。
