# IDEA-222 / Pairwise Local Likelihood-Ratio Verification (PLLRV)

- status: `rejected_at_gate1a`
- problem_category: `visual_representation`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-221 / SCLV`
- performance_status: `proof_of_path_failed`

## 路径

PLLRV固定S的Top-10候选，把V从绝对类别分数改成ordered pair局部似然比。对于A到B，使用private投影后的原型差方向在36patch上取证，奇函数聚合保证`L(A->B)=-L(B->A)`；I只选择最高挑战者并做OOF安全换位。

- old_solution_path: 为每个候选独立计算unary V，再在候选内argmax。
- new_solution_path: 对incumbent/challenger构造联合差方向 -> 局部成对证据 -> 反对称似然比 -> 安全二择一。
- principle_difference: 学习单位从单类别绝对相似度变成ordered candidate pair。
- non_equivalence_test: 必须胜同fold/同预算/同门控的unary-delta控制、随机和pair shuffle。
- minimal_viability: 500步OOF方向AUC、challenger命中、动作覆盖和反对称门。
- current_advantage: 精确反对称和可行动门成立，但pair方向与challenger判别失败。
- failure_boundary: mean8差向量平均掉细粒度部位差异；pair训练未能超过旧unary视觉。
- why_not_module: 新原语没有产生非平凡优于unary的行为，不能登记创新。
- paper_level_claim: none。

## 审核身份

- 最终Idea草稿SHA：`adacf22a281c1749ad37cb6213abb5f343b448b4e8cccacb7fbaa84c50c20636`；A/B最终审核SHA：`51b8d2b04a32c0b3e6116bae57813dff77b63e61af9f02124cd7ecc7820b64ff` / `9567d425e67a1806519ba13a58c666d18b1e2997b934ca03758eab5edcf671a6`，均P0/P1/P2=0。
- Gate1a脚本SHA：`5b2adb69df3aae93e2c41e686474327f82d33149478780561f15ad80e0683a31`；A/B交叉代码审查SHA：`d8c94b7c3ac1d39d54b313680a7438a3300b57ebdaea6994127fe30e1cd71924` / `5924e7dc61498223d0b558647801e6bca995ea6bc5dbfc9aec2da590fa81e940`，均P0/P1/P2=0。

## 500步OOF-only Gate1a

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/pllrv/IDEA-222-GATE1A/result.json@sha256:b73a611925a2b4372a51cf511cb722a3a9f2dda2078511f56dcc9fc828b9eac5`；日志SHA：`66aabc551713a900ef10a502f5f2cd068ce1ada1682fcc84df2d911e00882e15`。

- exact antisymmetry error=`0`，公式身份成立。
- direction AUC=`0.5641`，class-bootstrap lower=`0.4828`，未过`0.68/lower0.60`。
- 在1,904个`S-wrong && truth in C10`样本中，PairV challenger accuracy=`13.7605%`；随机=`11.1111%`，仅`+2.6494pp`；unary delta=`28.0462%`，PairV反而`-14.2857pp`。
- 正确challenger覆盖49类，低于50类门。
- Pair I自身OOF decisive AUC=`0.9038`，选`q=0.9`、coverage1.587%、恢复46/损坏40，说明后端仍能找到少量安全动作，但前端PairV方向本身不成立。
- pair/unary平均相关仅0.3147，不是简单复制unary；失败来自新原语更差，而不是等价。

因此不启动2,000步Gate1b/Gate2。下一补救保留pairwise验证，但不再先把8句角色文本压成mean8；各角色独立形成A/B差分视觉证据，再用交换不变权重做反对称聚合。

披露：Gate1a未加载official tensor，但研究路线继承已披露的official selection；`unseen_images_used_for_gradient=false`、`strict_blind_claim=false`。
