# FRAMEWORK-V5 Evidence Map

V5的唯一参数入口是`config/framework_v5.yaml`，结果表是`RESULTS.csv`，仓库外文件身份在
`ARTIFACTS.yaml`。历史事实继续引用原V4路径，不复制或重编号旧RUN。

## 核心闭环

1. `TUNE-002-RUN-030`提供TG+GTD Parent与152点Raw Off轨迹。
2. `V4-TRY-023-R2`提供固定训练checkpoint，真正unseen图像不进入梯度。
3. `V4-TRY-023-R3`固定candidate-local PCLR推理参数并达到H=`80.186419`。
4. `V4-TRY-023-R4`融合role6 overall appearance与role0 beak类别logits，正式H=`81.068777`。
5. Raw与R3 controls逐`U/S/H/ZS`复现source；双GPU metrics逐字节一致。

## 审核入口

- R1–R4单轮双Agent审核：`experiments/v4/innovation/V4-TRY-023-*_REVIEW.md`。
- R4最终审核：`experiments/v4/innovation/V4-TRY-023-R4_PCLR_SEMANTIC_ENSEMBLE_REVIEW.md`。
- 跨数据集审核：`experiments/v4/confirmation/V4-CONFIRM-003_PCLR_MULTIDATASET_REVIEW.md`。

## 声明边界

V5使用Chen-style official test选择checkpoint与推理超参数，且R3/R4属于嵌套test选择。
因此必须披露`test_used_for_selection=true`、`nested_official_test_selection=true`、
`strict_blind_claim=false`。CUB关系句使用LLM可见形态知识；AWA2/SUN确认使用generic类名方向句。
