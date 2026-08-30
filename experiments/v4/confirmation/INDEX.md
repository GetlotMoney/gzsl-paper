# FRAMEWORK-V4 Confirmation

V4不复制V3结果文件；以下证据直接绑定原RUN、原代码提交、原配置和仓库外输出。

| 数据集 | 原RUN | 条件 | U | S | H | ZS | 同checkpoint GTD增量H | 证据结论 |
|---|---|---|---:|---:|---:|---:|---:|---|
| CUB | V3-TRY-041 / control V3-TRY-040 | matched scratch fixed150 | 79.624420 | 76.670682 | 78.119641 | 85.794950 | +1.470620 | GTD正增益；owner接纳的本地matched确认 |
| AWA2 | V3-TRY-046 | scratch fixed150 | 97.553205 | 95.219404 | 96.372177 | 99.173242 | +0.000000 | 三数据集可运行；best点GTD为精确no-op边界 |
| SUN | V3-TRY-047 | scratch fixed150 | 80.208325 | 67.558140 | 73.341746 | 92.499995 | +3.099005 | GTD正增益，服务器两轮审查通过 |

## CUB证据

- TG-only：`V3-TRY-040`，code=`4d46ba1ef8d1c53c0e7fd5c5623f3c56af6dc1b2`，config=`config/tries/v3_try_040_tg_scratch_fixed150.yaml`，H=`76.649020`。
- TG+GTD：`V3-TRY-041`，同code与匹配初始化，config=`config/tries/v3_try_041_gtd_scratch_fixed150.yaml`，H=`78.119641`。
- config SHA：TRY-040=`b3b04dd71f188acc903b0b0e722eeeb027a15b731a88dee73362f3ffb7d3b469`；TRY-041=`4a7c4f7385d97a4c0294868cda9f8180eba9c53a55da6f76b27da94b47cd7e2b`。
- 输出：`D:/backup/Documents/ChatGPT/GZSL_Warehouse/tries/v3/gtd-scratch/V3-TRY-040`与`V3-TRY-041`。
- 原始结果账本：`experiments/v3/GTD_SCRATCH_CONFIRM_REVIEW.md`与`experiments/v3/EXPERIMENT_QUEUE.csv`。

## AWA2与SUN证据

- 最终RUN code：`4013cca894b00933f6bfed0a125690c66e54cba1`；两轮结论：P0/P1/P2=`0/0/0`，第2轮通过。
- AWA2：config=`config/tries/v3_try_046_gtd_awa2_scratch_fixed150.yaml`，config SHA=`44976e5e77a907112a88d295ee70296a2a22ef1c15c5b1f1d286249e1451a529`，asset=`7f0e1989635ca98d`。
- SUN：config=`config/tries/v3_try_047_gtd_sun_scratch_fixed150.yaml`，config SHA=`75a4035f783e92ba2cc70c3e7a633791abb80f3844a4f7b29ce1d172b4935cf1`，asset=`bfe12cda3c37abdb`。
- 输出：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v3/gtd-multidataset/V3-TRY-046`与`V3-TRY-047`。
- 共享审查证据：`/data/lby/projects/cv_project/GZSL_Warehouse/reviews/v3/gtd-multidataset/4013cca894b00933f6bfed0a125690c66e54cba1/shared_evidence.json`。
- 原始审查账本：`experiments/v3/GTD_MULTIDATASET_REVIEW.md`与`experiments/v3/EXPERIMENT_QUEUE.csv`。

## PCLR跨数据集确认（进行中）

| RUN | 数据集 | Source | 关系资产 | 预检ΔH | 状态 |
|---|---|---|---|---:|---|
| V4-CONFIRM-003-AWA2 | AWA2 | V3-TRY-046 | generic Top-3 / 117 edges | +0.233690 | audit passed; ready formal |
| V4-CONFIRM-003-SUN | SUN | V3-TRY-047 | generic Top-3 / 1633 edges | +0.983386 | audit passed; ready formal |

- 评估代码：`d1d8940844e2e540234a7cb8fae688d80efb5247`。
- AWA2 config SHA：`4c636b5371604cde8ae486ec09c81231d8b3580ce4a8d6a3a021227b4d083c88`；
  relation manifest SHA：`f93c9a690ce068614bc9792e6b60e989a4fe1fefebeb5df4a0273737e7bdadb2`。
- SUN config SHA：`43e5f568de8d998d0186116504527433d5c80194e67f527837ceea22ccadb356`；
  relation manifest SHA：`385744e7532ddf12862c3926878bac64422a902cdeeb87e30a9667045d4093b8`。
- 两套关系资产均为同CLIP类名方向句，`human_annotations_used=false`、
  `llm_world_knowledge_used=false`。它们验证可扩展的generic PCLR接口，不冒充CUB的LLM形态差异句。
- 正式签字：代码`d1d8940844e2e540234a7cb8fae688d80efb5247` + 审查tree
  `17e831fef00a09dbaa7882f61c7be13cb68b1790` + 双数据集config/asset SHA + 环境fingerprint
  `8b3e2d5d93cdd9763843c3c5f72903f466a86f7524c9dc2b02bb1d4699c32c59`。

## 适用边界

- 三数据集均验证同一TG+GTD代码路径、dataset-specific类别轴、U/S/H/ZS出口和theta-zero关闭路径。
- “三数据集验证”不等于“三数据集均由GTD提高H”：AWA2是明确的零增益边界，必须保留。
- 所有结果使用official test选择best-H，`test_used_for_selection=true`，不是blind-test。
