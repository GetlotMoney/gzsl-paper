# IDEA-223 / Role-Decomposed Pair Likelihood Verifier (R-PLLRV)

- status: `rejected_at_gate1a`
- problem_category: `visual_representation`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-222 / PLLRV`
- performance_status: `proof_of_path_failed`

## 路径与假设

R-PLLRV不先平均8句文本，而是对每个角色独立构造A/B差分patch证据；角色权重由交换不变的证据强度决定，最终保持严格反对称。它试图恢复mean8抹掉的稀疏细粒度差异。

- old_solution_path: 单一mean8 pair方向。
- new_solution_path: 8个对齐role差分场 -> 内容路由 -> 反对称pair likelihood。
- non_equivalence_test: 必须胜mean8 pair、unary delta、role shuffle/collapse/uniform。
- minimal_viability: 500步OOF方向、challenger、角色控制和action门。
- current_advantage: 无；角色路由近均匀且几乎还原mean8。
- failure_boundary: 共享投影与均匀角色聚合使多角色证据重新塌缩为平均方向。
- why_not_module: 所有核心视觉和角色非等价门失败。
- paper_level_claim: none。

## 身份与结果

- 最终Idea SHA：`c86217a231710053c8180d1734992e665b7cef4264ec70a02adfde9dc273e5bc`；A/B审核SHA：`beb4bb41a45280adfdc728e7b7c7a2af0beb379912118f9e229cec121b5fae2e` / `c89404b444bf956c026d572910962db058b4e7534dc65c9e09036468c113b59d`，均P0/P1/P2=0。
- 脚本SHA：`3eea7a366632d5ffb7c8d0d656650be5f10e3e8e13707af217039a2370b9fc92`；A/B交叉代码审查SHA：`4492aee1ab4c59d582242216591cf8aeefa6c90e05bda27bc88f0a9094a68468` / `77f745196cc2202bddcc759102e4206cd963d2a93ba89a81099b36611baf94b3`，均P0/P1/P2=0。
- 结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/rpllrv/IDEA-223-GATE1A/result.json@sha256:d266690b670a5a8b072108dc58ca779943750fea466375be070ceb594a3205a7`；日志SHA：`703385951581f34cb4465ce260591be3d6e4cd3cd9c11306a7f54e7bea09ce0b`。

## 500步OOF结果

- Full RolePair direction AUC=`0.5677`，bootstrap lower=`0.4857`；challenger accuracy=`12.4475%`，只比随机`11.1111%`高1.34pp。
- mean8 pair challenger=`13.7605%`，unary delta=`28.0462%`；RolePair均更差。
- role shuffle=`11.5546%`、uniform=`11.9748%`、frozen collapse=`12.4475%`，没有达到任何5pp非等价门。
- learned alpha entropy mean=`2.0774`，接近8角色均匀上限`ln(8)=2.0794`；alpha均值均约0.125。
- RolePair与retrained mean8 pair的候选排序相关均值=`0.9975`，实质塌缩回mean8。
- I仍能选`q=.9`、coverage1.885%、恢复53/损坏48，但Full OOF class-balanced accuracy=`71.2255%`，不胜unary的`71.2666%`，所有+1pp控制门失败。

下一补救放弃文本pair差向量，保留已验证更强的unary V，把视觉训练目标由全类CE改成S Top-10硬候选内的listwise/pairwise排序，直接对齐部署任务。

披露：未加载official；不使用教师、专家属性或unseen梯度。
