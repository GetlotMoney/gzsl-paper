# IDEA-224 / Hard-Candidate Listwise Unary Reranker (HCLR)

- status: `rejected_at_stage_a`
- problem_category: `candidate_verification`
- formal_parent_commit: `52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- rescue_of: `IDEA-223 / R-PLLRV`
- performance_status: `below_parent`
- innovation_claim: `none; borrowed learning-to-rank rescue module`

## 假设与结果

HCLR保留最强unary shared-coordinate V，把视觉训练目标从全类CE改成冻结S Top-10候选内的listwise CE＋pairwise hard-negative ranking，以对齐部署候选重排任务。

- old path: global all-class unary objective -> Top10 rerank。
- new module path: S Top10 -> candidate-only ranking gradient -> unary challenger -> safe I。
- minimal viability: 500步OOF先胜完整旧global unary，再训练额外因果控制。
- current advantage: 无；候选内排序显著破坏unary challenger能力。
- failure boundary: 在100个seen训练类上只优化局部候选list，丢失了global CE提供的跨类视觉坐标结构。

Idea最终草稿SHA：`695e4354c3316ad93bffad724b69e520351d864bfc58643e124d23d59b326e4b`；A/B复核SHA：`a00accbbefde6afe4e5937d14ca05763578d69f5424da4315b6972d20490036b` / `abd98dfaa7cd44252aeacb623aa1e7d0429055b77bc6b7e77bcd6f43e4d663b6f`。

StageA脚本SHA：`bc7d0b04e4a3b63a9a7edd88e200e6250ac4e77ceb51a2b4bde60b23b2032521`；A/B交叉代码审查SHA：`e42f95de9e72a036ef4a1efc374c42e29e129a4cd3c38091e5228d15c0f59ff2` / `11062ca1c8abc45c777bd5e084927f5576c95b1e8e9cc067b09472adebb09fe7`，均P0/P1/P2=0。

结果：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/hclr/IDEA-224-GATEA/result.json@sha256:14a0f8bf5d59fcb917d1df1664a1c1321022b8267c18de9b9efe47fcc31f147e`；日志SHA：`f0392b64d46aead77ba3ab9324f3196fa2c6080fc145b8818e6edd779882be2d`。

- HCLR challenger=`12.0273%`，只比随机`11.1111%`高0.916pp；global unary=`28.0462%`。
- HCLR true-vs-incumbent delta AUC=`0.5855`，lower=`0.5025`，未过门。
- I decisive AUC虽为`0.9359`，但最佳q=.95、coverage=.723%，低于1%门；恢复22/损坏15。
- OOF Full CBA=`71.2578%`，S=`71.1418%`，仅+0.116pp且bootstrap lower=-0.079pp；global控制因无合法q回退S。

结论：StageA失败，禁止继续StageB/2,000步。下一补救保留global unary的候选排序，不再单独改V；改为端到端的candidate utility reranker，让交互层联合使用S与V的每候选证据并以S残差方式安全排序。

披露：未加载official；不使用教师、专家属性或unseen梯度。
