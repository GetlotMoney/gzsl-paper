# V3-TRY-046/047 TG+GTD多数据集审查

- Owner指定base：`4d46ba1ef8d1c53c0e7fd5c5623f3c56af6dc1b2`。
- 初始冻结代码：`de5eb2dbea34bba35597b86aa31a84d9a852c1b9`。
- 集中修复后最终RUN代码：`4013cca894b00933f6bfed0a125690c66e54cba1`。
- AWA2 config SHA：`44976e5e77a907112a88d295ee70296a2a22ef1c15c5b1f1d286249e1451a529`。
- SUN config SHA：`75a4035f783e92ba2cc70c3e7a633791abb80f3844a4f7b29ce1d172b4935cf1`。
- 共享证据：`/data/lby/projects/cv_project/GZSL_Warehouse/reviews/v3/gtd-multidataset/4013cca894b00933f6bfed0a125690c66e54cba1/shared_evidence.json`。
- 共享证据SHA：`d0b851f148da271ca35d296cb375de5dd45677b571a73213bc7aa99f5df812a4`。
- 本地专项：`13 passed`；本地全量：`536 passed, 2既有warnings, 3 subtests passed`。
- 服务器真实证据：AWA2=`cuda:0`、SUN=`cuda:1`；两条路径的真实资产shape/split/dtype/有限性、parent与Gate非零梯度、U/S/H/ZS、关闭路径及checkpoint roundtrip均通过。

## 缺陷与集中修复

- 初始Round 1与Round 2预读均发现同一P1：scratch配置的`parent_metrics_percent=null`，resume未恢复首次运行计算出的update-0指标锚点，下一评估会崩。
- 集中修复`4013cca`：新checkpoint显式保存四指标父锚点；resume严格恢复并校验有限性、字段完整性和warm-start配置一致性；旧checkpoint可从`history[0] / update=0`回退恢复。
- 修复专项和服务器证据均验证显式字段、旧history fallback以及下一评估delta计算。

## 最终结论

- Round 1：`4013cca / P0=0 / P1=0 / P2=0 / 第1轮通过`。
- Round 2：`4013cca / server HEAD准确且clean / P0=0 / P1=0 / P2=0 / 无P0/P1，第2轮通过`。
- 两条RUN只允许在准确、干净的`4013cca`执行；配置、资产、代码或计算语义变化会使签字失效。
