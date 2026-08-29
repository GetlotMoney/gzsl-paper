# IDEA-171：Hypothesis-Conditioned Visual Completion（HCVC，假设条件视觉补全）

idea_id: IDEA-171
source_type: experiment_result + first_principles + owner_hypothesis + nearest_work_boundary
status: proposed_owner_confirmed_proof_of_path_candidate
problem_category: visual_grounding
mechanism_tags: [class_conditioned_completion, masked_visual_prediction, contextual_region_token, conditional_energy, candidate_hypothesis_verification, class_disjoint_transfer]
base_framework: FRAMEWORK-V4
base_commit: 52088f69d7ac4e574e7b63c28b21ac0da7789933
reuse_refs: [IDEA-162, IDEA-163, IDEA-164, IDEA-165, IDEA-166, IDEA-167, IDEA-168, IDEA-169, IDEA-170]
method_name: Hypothesis-Conditioned Visual Completion
method_acronym: HCVC
chinese_name: 假设条件视觉补全

## 1. 要解决的问题

V4的TG+GTD最终仍把类别表示为等待与图像比较的文本点原型。已经运行并失败的IDEA-163、164、165、166、168和169表明，继续使用patch-text静态相似度、Attention聚合、局部重排、三态证据、图搜索或事后输入干预，没有形成稳定任务增益；IDEA-167与IDEA-170仍处于pending，不能当作失败证据。

IDEA-162只证明了一件更窄的事：同一冻结CLIP最终层patch经过跨类别共享学习型Reader后，在100类训练、50类隔离条件下，概念median AUC由`0.589784`升到`0.774050`，打乱标签为`0.482730`。它支持“视觉token中存在可迁移的可读信号”，但不证明空间定位、masked completion、候选条件能量或任务重排有效。

HCVC要检验的新问题是：

> 若正确类别真能解释当前图像，那么在验证器没有看到某一区域原像素时，正确类别文本是否能结合剩余视觉上下文，比错误类别更准确地预测该区域的上下文化视觉语义？

## 2. 可证伪假设

只用100个开发seen类图像训练共享补全器，并让50个class-disjoint开发unseen类图像完全不进入TG+GTD父模型或HCVC的梯度、checkpoint选择和统计拟合。若正确的TG+GTD候选文本确实提供当前实例特有的视觉预测信息，则在50类图像上：

1. 真类条件补全能量应低于Top-5内最难错误候选；
2. 以条件补全能量重选同一冻结TG+GTD Top-5，应提高逐类Top-1；
3. 收益不能被同容量纯排序器、text-only预测、类别平均残差或泄漏路径解释。

Gate 0只回答`class-disjoint unseen-candidate proof-of-path`，不报告official GZSL U/S/H/ZS，也不证明第三核心创新已经成立。

## 3. 范式准入字段

old_signal_or_primitive: 类别是单点文本原型；视觉证据是图像或patch与文本的静态相似度、聚合分数或事后干预响应。

new_signal_or_primitive: 新增训练信号是输入级遮挡后的seen图像区域预测；新操作对象是类别条件预测能量`E(r_m | masked_context, t_c, position_m)`。类别文本不只参与匹配，还参与预测验证器未见原像素所对应的上下文化region-token residual。

paradigm_shift: 相对TG+GTD父框架，候选判别由“哪个点原型与图像最相似”改写为“哪个候选条件模型最能预测当前实例被留出的视觉证据”。这属于父框架没有的masked预测学习信号；不声称生成式分类、类别条件分布或masked reconstruction本身是新范式。

why_not_module: rank-64、MLP和位置向量只是Core的实现工具；margin只属于旁报的HCVC+Rank，不是范式核心。范式候选成立必须依赖输入CLIP前的真实遮挡目标、类别条件区域预测及target-shuffle反证；若收益可由同容量ranking-only控制复现，或同类别跨图target shuffle不使收益坍塌，则HCVC退化为普通重排器或类别均值记忆，立即drop。

paper_level_claim: 只有Gate 0、后续正式class-disjoint GZSL实验和多seed均成立后，最多窄化声称：“我们把既有生成式分类原则具体化为class-disjoint GZSL候选验证机制：共享轻量补全器在输入级遮挡的冻结CLIP空间中预测上下文化region-token residual，并利用masked visual context与类别文本的交互能量，在不使用pseudo-unseen图像训练的情况下验证冻结TG+GTD Top-K候选。”不得声称首次类别预测假设、首次生成式零样本分类、首次类别条件inpainting验证、首次条件分布或首次masked语义预测。

## 4. 最近工作与原创边界

- [Diffusion Classifier（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html)已经用候选prompt条件噪声预测误差近似类别条件概率并做零样本分类。因此“按候选预测误差分类”不是HCVC的新颖点；差异只能收窄到class-disjoint GZSL中的冻结TG+GTD Top-K、输入级masked CLIP上下文、共享轻量region-token residual补全器及无pseudo-unseen图像训练。
- [RONIN（WACV 2026）](https://openaccess.thecvf.com/content/WACV2026/papers/Nguyen_Detecting_Out-of-Distribution_Objects_through_Class-Conditioned_Inpainting_WACV_2026_paper.pdf)已经用预测类别条件inpainting及原图—补全一致性验证预测。因此“类别假设通过补全被验证”也不是新颖点；HCVC必须把claim限制在上述GZSL数据边界和冻结CLIP区域语义能量的具体机制。
- [Class-Conditioned Deep Generative Models for ZSL](https://arxiv.org/abs/1711.05820)、[f-CLSWGAN（CVPR 2018）](https://arxiv.org/abs/1712.00981)和[VADS（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Visual-Augmented_Dynamic_Semantic_Prototype_for_Generative_Zero-Shot_Learning_CVPR_2024_paper.html)已经把类别语义用于条件分布或视觉特征生成。因此“类别从点变为条件分布”不能单独作为原创claim；HCVC不生成unseen伪样本，而对当前真实图像被留出的区域证据计算候选条件能量。
- [MAE（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)、[M3AE](https://openreview.net/forum?id=qmyvfCPnx-e)和[RILS（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_RILS_Masked_Visual_Reconstruction_in_Language_Semantic_Space_CVPR_2023_paper.html)分别覆盖masked视觉重建、多模态masked学习和语言语义空间重建；HCVC不主张这些学习信号本身新颖。
- [ConMIM（Findings of EMNLP 2022）](https://aclanthology.org/2022.findings-emnlp.10/)已使用文本信息辅助masked image reconstruction，是邻近学习信号先例，但没有枚举候选类别文本进行生成式分类。
- [Conditional Visual Classification（ICCV 2019）](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Rethinking_Zero-Shot_Learning_A_Conditional_Visual_Classification_Perspective_ICCV_2019_paper.html)已按语义生成条件分类器；HCVC的条件对象是当前实例被留出的region-token residual，而不是分类器权重。

closest_paradigm_work: Diffusion Classifier和RONIN是最接近的范式边界；RILS、M3AE、ConMIM及生成式ZSL是机制或表示边界。

closest_work_conclusion: 截至2026-08-30的定向原文核对，类别条件预测误差分类、类别条件inpainting验证、masked语义重建和生成式ZSL均有明确先例。当前只保留上述窄机制组合的候选新颖性；这不是系统检索完成，不得写“首次”。

## 5. Gate 0唯一数据边界

### 5.1 固定开发划分

- 数据仅为CUB xlsa17 `trainval_loc`中的150个formal-seen类；official 50个test-unseen类及图像不加载。
- 采用xlsa17标准开发类划分：`train_loc`的100类/4,702张图为`dev-seen`，`val_loc`的50类/2,355张图为`dev-unseen`。
- 现有split schema为`gzsl-paper.cub-standard-validation.v1`，`split_seed=20260823`；类别列表规范化JSON SHA256为`95d5abbd5dd56035e567bc55c97a4604d680ab3f3400a15901c91b9a29a2d4ac`。
- 100个dev-seen全局类ID固定为：`[0,1,4,5,7,8,9,10,12,13,14,15,16,17,19,21,22,23,24,25,26,27,29,30,32,34,37,38,40,42,45,46,47,50,53,56,58,59,60,62,64,65,66,69,72,73,74,75,77,80,84,85,88,89,92,93,95,96,98,102,105,106,108,111,113,117,118,122,125,126,127,130,131,133,135,143,146,148,150,152,153,155,161,163,164,168,171,176,177,179,180,182,187,189,193,195,196,197,198,199]`。
- 50个dev-unseen全局类ID固定为：`[2,3,11,31,36,39,41,43,44,48,51,52,54,57,63,70,76,81,82,83,91,100,101,104,109,110,112,114,116,120,129,132,134,136,137,139,142,144,145,147,154,157,160,162,167,169,172,174,183,185]`。

### 5.1.1 唯一global↔local类别轴

xlsa全局类别ID位于`0..199`且当前150类不连续，禁止直接用global ID索引150行tensor。映射唯一固定为：

```text
active_global = sort(dev_seen_global union dev_unseen_global)
local_to_global[150] = active_global
global_to_local[200] = -1
global_to_local[active_global] = arange(150)
```

`active_global`必须逐值等于上一节两张类表并集升序，其compact JSON SHA256=`aaee779ba7fbb0908ec1839c990e4523defe5832fa5f4b3e840d4557f8c99f42`；local `0..149` compact JSON SHA256=`1a7b80b181100aba628ccce7ba02bab13462893ae11015f7d9e72184bacbfeca`。mapping receipt必须保存`active_global/local_to_global/global_to_local`逐项tensor与JSON、两种SHA及算法版本。

- `role_text[150,8,768]`严格按`local_to_global`从源200类文本做确定性`index_select`后编码；模型内部train/eval labels均保存local `0..149`。
- Parent seen classes固定为`global_to_local[dev_seen_global]`的100个local ID，GTD dev-unseen固定为`global_to_local[dev_unseen_global]`的50个local ID；topology内部使用完整local `0..149`，CE/visual-centroid/GTD teacher只使用100个local seen轴。
- Parent logits、Top-1/Top-5、候选`t_c`、hard negative、correction/damage和per-class receipt必须同时保存local与global ID；所有稳定排序、tie-break、统计和最终报告只按global ID。
- Block-text及其他类别mapping先在global ID域按第7节规则生成，进入tensor前统一通过`global_to_local`转换；任何代码直接用global ID索引150行tensor立即报错。
- Official 50类test-unseen图像/标签绝不加载，official 50类文本不得进入Gate tensor、模型、训练、候选或统计。生成器若读取冻结200类role JSON，只允许执行无拟合的`index_select(active_global)`；manifest必须记录200类source JSON SHA、selected 150 IDs、mapping SHA及输出`role_text[150]` SHA，禁止生成或暴露official 50类文本embedding。

### 5.1.2 local类别ID到100列CE位置

100个dev-seen local ID在`0..149`中仍不连续，不能直接作为100列seen logits的CrossEntropy target。第三层且最后一层类别映射唯一固定为：

```text
seen_local = sort(global_to_local[dev_seen_global])
local_to_seen_position[150] = -1
local_to_seen_position[seen_local] = arange(100)
ce_target = local_to_seen_position[train_local_label]
assert 0 <= ce_target < 100
```

`seen_local` compact JSON SHA256=`b3c2777870bad18210a155b97ebf56e5848d59ca6b1fa26018e3c26c9ecfcffd`；`local_to_seen_position` compact JSON SHA256=`52548ceeccb701e4bcbd6984c02d0a69b28f99b09d77959cdd0cc5eab4bbb8d0`。

- `parent.logits(images, seen_local)`的第`j`列严格对应`seen_local[j]`；`visual_centroids[100,768]`第`j`行也严格对应`seen_local[j]`。
- GTD folds、pseudo-seen/pseudo-unseen和teacher始终使用local class ID，不得使用CE position。
- Hard negative转换链固定为`100列position → seen_local[position] → local_to_global[local_id] → 双ID receipt`。
- 150类联合评估logits第`j`列固定对应local ID `j`，再经`local_to_global[j]`回写global ID；HCVC filtered labels继续保存local ID，类别mapping仍在global域生成后转local。
- 任何local label未经`local_to_seen_position`直接送入100列CE立即报错。receipt必须保存`seen_local/local_to_seen_position`的tensor、compact JSON和SHA，并绑定seen-logit列顺序SHA与visual-centroid行顺序SHA；本Gate不存在第四种类别轴。

### 5.2 无泄漏TG+GTD父模型

Gate 0禁止复用任何在150类formal-seen图像上训练过的V4 checkpoint。必须先生成一个独立parent receipt/checkpoint，并满足：

- 代码仍继承`FRAMEWORK-V4@52088f69d7ac4e574e7b63c28b21ac0da7789933`的TG+GTD计算语义；
- TG+GTD梯度只读取4,702张dev-seen图像和其100类标签；50类dev-unseen图像不加载、不进入梯度、评估或checkpoint选择；
- 150类冻结文本均可作为标准ZSL候选表达。TG topology loss固定weight=`0.1`并在完整150类冻结文本原型轴上计算；50个dev-unseen类可参与这一无监督文本拓扑正则并作为GTD待迁移候选，但禁止提供图像、标签、视觉中心或GTD teacher。CE class axis、visual-centroid class axis和GTD teacher class axis都只允许100个dev-seen类；
- 该parent诚实标记为Gate专用`fixed-50 diagnostic parent`，不是V4 fixed-150正式训练，也不声称“只改变数据”。固定seed7、batch size 50、50 nominal epochs、`updates_per_epoch=floor(4702/50)=94`、总计4,700 updates；每步由独立CPU generator `manual_seed(7)`执行`torch.randperm(4702, generator=parent_cpu_generator)[:50]`，batch内无放回、跨step重新排列；
- optimizer固定Adam，betas=`(0.9,0.999)`、eps=`1e-8`、weight decay=`1e-4`；TG lr/min=`1e-4/1e-4`，Gate lr/min=`1e-4/1e-5`，scheduler horizon=4,700 updates，gate warmup=`5×94=470` updates；
- 其他父参数逐值固定：topology loss weight=`0.1`、gate loss weight=`1.0`、dropout=`0.5`、TG inner/outer ratio=`0.35/0.65`、temperature=`0.07`、GTD hidden dim=`16`、geodesic grid points=`33`、theta penalty=`0.1`、max transport step=`1.5`、dead zone=`1/(33-1)=0.03125`；
- teacher refresh固定在`1+94k, k=0..49`。parent初始化由seed7生成并在receipt记录initial state SHA、batch轨迹SHA和refresh轨迹；禁止dev-unseen评估、early stop或best选择，只保存update 4,700最后状态；
- parent固定strict deterministic：首次CUDA初始化前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`，随后设置`torch.backends.cuda.matmul.allow_tf32=False`、`torch.backends.cudnn.allow_tf32=False`、`torch.backends.cudnn.deterministic=True`、`torch.backends.cudnn.benchmark=False`和`torch.use_deterministic_algorithms(True)`；receipt必须记录每个开关的实际生效值、Python/PyTorch/CUDA/cuDNN/GPU及相关环境；
- receipt必须绑定准确code commit、config SHA、split SHA、图像/文本资产manifest SHA、seed、停止update及checkpoint SHA。
- parent receipt必须记录完整150个local `topology_class_ids=0..149`及SHA，同时记录映射后的global active轴SHA256=`aaee779ba7fbb0908ec1839c990e4523defe5832fa5f4b3e840d4557f8c99f42`，并分别记录`CE/visual-centroid/GTD-teacher`三个100类local/global轴及SHA。

Gate 0的baseline Top-1、150类联合竞争Top-5、候选表示`t_c`和训练困难负类必须全部来自这一准确冻结TG+GTD parent。`t_c`固定为parent输出的768维最终类别原型，不再混用TG-only或原始local/unique文本。

Parent Top-1、Top-5及全部candidate receipt统一按`(-logit, global_class_id)`稳定排序，完全平局取较小全局class ID，禁止依赖`topk`的未声明平局顺序。HCVC+Rank与Ranking-only必须读取同一份冻结hard-negative receipt及SHA；该receipt在100个dev-seen候选中按上述稳定排序为每个`N_train`图像选择最高错误类，不得由两个条件分别重算。

HCVC训练时，真实文本及困难负文本只来自100个dev-seen类；50个dev-unseen文本可以由冻结parent在评估时表达候选，但不进入HCVC梯度。

### 5.3 100/50物理资产与进程隔离

禁止任何训练进程打开物理包含全部7,057张图像特征的旧共享tensor后再切片。第6节的同一个新FP32 CLIP生成器、同checkpoint、同预处理和同compute identity必须一次性产生两套分层manifest：

- `parent_train_manifest`只包含`dev_seen_cls_features[4702,768]` FP32、local `dev_seen_labels[4702]`、按`local_to_global`排列的`role_text[150,8,768]` FP32和train raw region caches；
- `parent_eval_manifest`独立包含`dev_unseen_cls_features[2355,768]` FP32、local `dev_unseen_labels[2355]`和eval raw region caches。

全局CLS、类别文本、teacher region target和masked-student context必须来自同一生成器与checkpoint空间。Parent训练loader只能打开train manifest允许键并记录实际读取键；parent冻结后，独立只读parent-eval进程才可打开eval manifest生成Top-1/Top-5。HCVC训练loader只能打开filtered `N_train`；HCVC冻结后，独立eval进程才可打开filtered `N_eval`。任一训练loader出现eval feature/label键立即拒绝。两套manifest共享生成器receipt并各自记录上游SHA、键、shape、dtype和tensor SHA。

Filtered train manifest允许键唯一为`relative_paths[N_train]`、local `labels[N_train]`（只含100个dev-seen local IDs）、`context[N_train,4,432,768]` FP16和`target[N_train,4,768]` FP32，并独立绑定类别mapping SHA、parent checkpoint SHA、150类prototype tensor SHA与hard-negative receipt SHA。Filtered eval manifest允许键唯一为同构的`relative_paths[N_eval]`、local `labels[N_eval]`（只含50个dev-unseen local IDs）、`context[N_eval,4,432,768]` FP16和`target[N_eval,4,768]` FP32，并独立绑定类别mapping SHA、parent checkpoint SHA与Top-5 candidate receipt SHA。两个manifest都记录来源池count、有效count、local/global class IDs和allowed keys；训练loader出现任何eval path/label/context/target直接拒绝，eval manifest只能在模型冻结后打开。

## 6. 唯一视觉与补全合同

### 6.1 CLIP与遮挡

- 冻结OpenAI CLIP ViT-L/14@336，checkpoint固定为`/data/lby/projects/cv_project/GZSL_Warehouse/assets/clip_checkpoints/ViT-L-14-336px.pt@sha256:3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`，来源合同为`config/paper_v2/DATA_SOURCES.yaml`与`config/paper_v2/ASSETS.yaml`。teacher和student共享同一冻结权重，并由同一个新缓存生成器在同一次身份合同下产生。禁止复用旧FP16 teacher patch再转成FP32充当本Gate target。
- 缓存计算路径唯一固定为：首次CUDA初始化前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`；随后执行`model.float()`、normalized input FP32、`model.eval()`、`torch.inference_mode()`、禁止autocast、`torch.backends.cuda.matmul.allow_tf32=False`、`torch.backends.cudnn.allow_tf32=False`、`torch.use_deterministic_algorithms(True)`、cuDNN deterministic=true/benchmark=false、`cuda:0`、batch16。projected token在GPU保持FP32。
- 图像预处理固定为OpenAI CLIP的RGB、bicubic resize、336×336 center crop、tensor和CLIP mean/std normalize。
- ViT-L/14@336形成24×24 patch网格。四个mask固定为四象限：`[0:12,0:12]`、`[0:12,12:24]`、`[12:24,0:12]`、`[12:24,12:24]`。
- 每次只mask一个象限。mask在normalize后、patch embedding前执行：对应像素置0；这严格等价于normalize前填充CLIP RGB mean，文档和代码不得再提供其他fill路径。
- 完整图像只进入teacher；每个mask图像单独进入student。decoder只能读取masked student的432个非目标位置token；目标位置student token、完整teacher CLS、完整teacher非目标token均不得进入context。
- teacher/student token均固定取visual transformer final block之后，对全部token应用`ln_post`，再乘`visual.proj`得到768维表示。CLIP projected token立即cast FP32；由于这些token已经经过全局自注意力，统一称`contextualized region token`，不得称独立local patch证据。

### 6.2 目标残差

对图像`i`、mask象限`m`：

```text
u_i,m = mean(teacher_projected_tokens[target quadrant m])
g_i   = normalize(teacher_projected_CLS, eps=1e-6)
r_raw_i,m = u_i,m - dot(u_i,m, g_i) * g_i
r_i,m = normalize(r_raw_i,m, eps=1e-6)
```

`teacher_projected_CLS`和teacher目标token只用于构造stop-gradient target。teacher residual、norm和无效判定全部用FP32；target residual以FP32缓存。masked-student visible context以FP16缓存，读入decoder后立即cast FP32；decoder、energy和loss全部FP32。若任一区域`||r_raw||<1e-6`，或任一masked view满足`||mean(normalize(student_visible_tokens, eps=1e-6))||<1e-6`，该图像标记无效并从全部条件共同排除；两类无效原因分别计数。train与eval来源池无效率分别计算为`train_invalid/4702`和`eval_invalid/2355`，两者都必须≤1%，任一超限即Gate失败，禁止合并分母。必须报告四区域原始范数分布、visible-pool范数分布、近零数和epsilon触发率。

### 6.3 共享rank-64补全器

对masked student的432个非目标token先逐tokenL2归一化再均值池化：

```text
v_i,m = normalize(mean(normalize(student_visible_tokens, eps=1e-6)), eps=1e-6)
a_i,m = W_v v_i,m
b_c   = W_t normalize(t_c, eps=1e-6)
h_i,c,m = GELU(a_i,m + b_c + a_i,m * b_c + e_m)
r_hat_i,c,m = normalize(W_o h_i,c,m, eps=1e-6)
```

其中`W_v:768→64`、`W_t:768→64`、`W_o:64→768`均无bias，`e_m`是固定四槽可学习位置表`4×64`；`*`表示逐元素乘法。总参数固定为`147,712`，不增加层、dropout、LayerNorm、Attention或候选专属参数。

初始化唯一固定为：在CPU上先`torch.manual_seed(7)`，依次以`nn.Linear(..., bias=False)`构造`W_v/W_t/W_o`并保留PyTorch默认`reset_parameters`，`e_m=torch.zeros(4,64)`；保存一次initial state及SHA，所有可训练条件从该状态逐值clone后再移到GPU。禁止为不同条件重新随机初始化。

候选条件能量：

```text
E_i,c = (1/4) * sum_m ||r_i,m - r_hat_i,c,m||_2^2
```

主分类分数固定为`-E_i,c`，Top-5内取最小能量；不再称Gaussian likelihood或log-likelihood，不使用`E_image_only-E_c`排序。context-only（原草稿image-only）对同一图像的各候选是常数，只作绝对误差诊断。

### 6.4 训练目标与停止

- 4,702张dev-seen图只是训练来源池。训练前仅依据候选/类别无关的teacher residual与visible-pool数值规则得到固定`N_train`有效索引；所有条件复用完全相同的有效索引、initial state和batch轨迹。2,355张dev-unseen来源池同样先得到固定`N_eval`；任一类没有有效图像直接Gate失败。有效索引receipt必须绑定规则、索引顺序SHA、`N_train/N_eval`及两类过滤原因。
- 每步固定由seed7 CPU generator执行`torch.randperm(N_train, generator=cpu_generator)[:16]`：batch内无放回、跨step重新排列；四个区域全部计入loss。
- `HCVC-Core`是唯一范式主条件，只训练正样本补全：`L_core=L_completion=mean(E_i,y)`，不读取负候选标签。
- `HCVC+Rank`仅作普通辅助模块诊断。其困难负类固定为冻结parent在100个dev-seen类中logit最高的错误类，logit平局取全局class ID较小者；`L=L_completion+L_rank`，`L_rank=mean(max(0,0.1+E_i,y-E_i,n))`，两项权重均为1、margin固定0.1。它不能救活失败的Core，也不能在看到Gate结果后被后选为paper-level核心。
- AdamW固定lr=`1e-3`、weight decay=`1e-4`、betas=`(0.9,0.999)`、eps=`1e-8`、amsgrad=false、constant LR、无scheduler、无gradient clipping；固定1,000 updates并保存最后状态，不加载dev-unseen结果选checkpoint。
- 所有能量平局均取全局class ID较小者；任一`t_c`原始范数小于`1e-6`、NaN/Inf、预测范数小于`1e-6`或缺少任一区域缓存均直接报错，不静默替换。

HCVC-Core及所有可训练控制和energy评估进程另行固定运行入口；不得假设parent或缓存进程的环境会自动继承：首次CUDA初始化前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`，设备固定`cuda:0`，执行`torch.manual_seed(7)`与`torch.cuda.manual_seed_all(7)`，并设置`torch.backends.cuda.matmul.allow_tf32=False`、`torch.backends.cudnn.allow_tf32=False`、cuDNN deterministic=true/benchmark=false及`torch.use_deterministic_algorithms(True)`。AdamW还显式固定`foreach=False`、`fused=False`、`maximize=False`、`capturable=False`、`differentiable=False`。运行receipt必须记录全部实际生效值、Python/PyTorch/CUDA/cuDNN/GPU、initial-state SHA、batch轨迹SHA和final-state SHA。

## 7. 必要对照及不可替代条件

所有可训练对照使用相同split、seed、1,000 updates、batch16、优化器、rank64和参数预算；除明确关闭项外共享数据轨迹。

1. `Parent`：冻结TG+GTD Top-1，不运行补全器。
2. `HCVC-Core`：第6节正样本补全MSE，是唯一范式主条件。
3. `HCVC+Rank`：在Core上加入negative ranking，只旁报相对Core的差值与CI；不能参与范式救活或后选。
4. `Ranking-only`：保留完全相同的`W_v/W_t/W_o/e_m`和`h`，不读取teacher target；以`q=W_o h`、`s_i,c=mean_m ||q_i,c,m||²`作为候选能量，固定训练`L_rank=mean(max(0,0.1+s_i,y-s_i,n))`，其中`n`只能读取与HCVC+Rank共享的冻结hard-negative receipt，禁止重算。其能量尺度自由，只比较分类结果，不与Core或Context-only比较能量绝对值。
5. `Text-only`：与Core相同，但`a_i,m=0`，只由候选文本和mask位置预测真实target。
6. `Context-only/null-text`：与Core相同，但`b_c=0`且去掉`a*b`；只报告真实target MSE，候选轴恒定，不报告分类提升。
7. `Shuffled-text`：使用block-diagonal单循环。两个类别块分别按global class ID升序；单一CPU generator `manual_seed(1007)`依次执行`randperm(100)`和`randperm(50)`，每块对排列`p`定义`p[j]→p[(j+1) mod n]`。训练只读取100类块，评估使用两块共同组成的150类错误映射。
8. `Same-class target shuffle`主控：只在训练时保持当前图像context、真实类别和mask位置，但换成同类别、同mask、不同原始图像target。global class ID升序；每类有效图按相对路径升序；单一CPU generator `manual_seed(2007)`按类顺序依次执行`randperm(n)`，并定义`p[j]→p[(j+1) mod n]`。生成前每个dev-seen类必须至少有2张有效训练图，否则Gate失败。评估一律恢复真实、未shuffle的teacher target。
9. `Global target shuffle`旁报：全部有效训练图按相对路径升序；CPU generator `manual_seed(3007)`执行`randperm(N_train)`并定义`p[j]→p[(j+1) mod N_train]`，只改变训练target且保持mask位置；评估一律使用真实teacher target。
10. `Post-encoding mask`泄漏上界：完整图像先经CLIP再删除目标token，使用与Core相同的初始化、有效索引、batch轨迹、1,000 updates和优化器独立训练；只证明泄漏能达到的上界，不可替代主路径。禁止只在评估阶段替换context造成分布错配。

上述三类mapping都只能在valid-index receipt冻结后生成；各自mapping receipt必须绑定valid receipt SHA、算法版本、seed、排序规则、完整source→target数组及mapping SHA，禁止实现者另选derangement算法。

HCVC-Core必须同时超过Parent、Ranking-only和Text-only，并通过target-shuffle及Context-only误差门；否则不能把收益归因于“当前实例上下文×候选文本的留出证据补全”。

## 8. Gate 0评估与统计合同

- 评估来源池为2,355张dev-unseen图像，主结果只使用预先冻结的共同有效索引`N_eval`；候选是无泄漏parent在100 dev-seen + 50 dev-unseen的150类联合竞争Top-5。2,355来源池上的Parent结果只旁报，禁止与`N_eval`上的Core混用分母。
- 主任务指标是`N_eval`上50个dev-unseen类的macro Top-1；同时报告同一分母的micro Top-1、每类Top-1、逐图correction/damage表。coverage、oracle ceiling、McNemar及所有对照差值也只使用`N_eval`。
- 真类不在Top-5时按无法纠正计入全部样本主结果；covered子集另报但不得替代全部样本。
- `hard wrong`固定为Top-5错误候选中HCVC-Core能量最低者。对每个dev-unseen类`k`，只在该类true进入Top-5的有效图上计算`rate_k=mean(E_true<E_hard_wrong)`；任一类`covered_count_k=0`立即判`parent_candidate_failed`。主观察值为50类`rate_k`的macro mean，冻结类别bootstrap矩阵复用于这一50维向量；全部covered图像的micro rate只旁报。
- correction=`Parent错→Core对`；damage=`Parent对→Core错`；net correction=`correction-damage`。
- 报告Top-5 macro/micro coverage、`covered-but-Top1-wrong`数量及oracle ceiling。Top-5 macro coverage必须≥80%，且`covered-but-Top1-wrong`必须≥40；否则Gate因父候选能力不足失败，不调K。
- 统计前按global class ID升序形成paired per-class向量。单一CPU generator `manual_seed(7)`一次生成`[10000,50]` int64 bootstrap index matrix：每行调用`torch.randint(0,50,(50,),generator=bootstrap_generator)`有放回采样；所有macro差值、能量排序率和relative-MSE共用该矩阵。输入向量与bootstrap均在CPU FP64计算，CI固定为`torch.quantile(samples,[0.025,0.975],interpolation="linear")`；统计receipt保存matrix、matrix SHA、类别顺序和PyTorch版本。
- Parent与HCVC-Core在全`N_eval`上执行exact two-sided McNemar：`b=Parent错/Core对`、`c=Parent对/Core错`；若`b+c=0`则`p=1`，否则`p=min(1,2*sum(comb(b+c,k),k=0..min(b,c))/2^(b+c))`。规范实现使用Python整数`math.comb`后转FP64，receipt记录输入`b/c`、Python版本与结果；禁止改用连续性校正或渐近卡方。

Gate 0只有以下条件全部满足才通过：

1. HCVC-Core的50类macro `mean_k(rate_k)`观察值≥60%，且类别bootstrap 95%下界>0.5；covered图像micro rate不进入门槛；
2. HCVC-Core相对Parent的macro Top-1至少`+1.0pp`，类别bootstrap 95%下界>0，exact McNemar `p<0.05`，correction≥20且net correction≥20；
3. HCVC-Core分别超过Ranking-only和Text-only至少`+0.5pp`，两项类别bootstrap差值95%下界均>0；
4. Shuffled-text和Same-class target shuffle相对Parent的正增益各自不超过Core正增益的20%，且各自net correction<20；Global target shuffle只旁报；
5. 对每个dev-unseen类先分别计算Context-only与Core的图像平均真实target MSE；任一类`MSE_context_only<=1e-12`直接判数值失败，禁止临时加epsilon。其余类别计算`relative_mse_drop=(MSE_context_only-MSE_core)/MSE_context_only`；主值取50类relative drop的macro mean。类别bootstrap每次重采50类并重算macro mean；主值必须≥5%，95%下界>0；
6. 无效率≤1%，所有缓存、能量和输出finite，主路径没有读取teacher或目标位置student token。

`HCVC+Rank`只报告相对Core的macro Top-1差值、类别bootstrap CI、correction/damage和额外训练依赖；它不进入上述通过门，也不能替代失败的Core。

任一门失败立即判定`proof_of_path_failed`并drop；不得修改mask数/位置/fill、CLIP层、rank、decoder结构、margin、loss权重、训练步数、Top-K、候选轴或原logit融合进行补救。

## 9. 非等价与泄漏判定

- Ranking-only达到与Core相同收益：说明新target学习信号不是必要原因，HCVC退化为普通判别重排，失败。
- Same-class target shuffle仍保留超过20%的Core增益：说明模型主要记忆类别平均残差，而非当前实例留出证据，失败。
- Text-only与Core差距不足0.5pp：说明当前图像context没有必要作用，失败。
- Post-encoding mask显著优于输入前遮挡只能说明完整token泄漏；它不能成为主结果。
- Context-only的候选分数完全相同是数学必然，只作为误差诊断，不得包装成分类基线。

## 10. 复杂度合同

- 每张图总视觉成本为1次完整teacher forward + 4次masked student forward；若完整teacher资产可按准确checkpoint/预处理/层身份复用，必须同时报告总成本和相对Parent新增的4次forward。
- 四个masked context与teacher target离线缓存，候选Top-5不重复运行CLIP；共享context一次编码后向量化计算`5 candidates × 4 masks`。
- 缓存身份固定为无循环三段链：`raw source-cache manifest → valid-index receipt → filtered/final cache manifest`。raw manifest覆盖4,702/2,355两个完整来源池及用于有效性判定的raw residual/visible-pool统计；valid receipt只绑定raw manifest SHA并产出`N_train/N_eval`索引；filtered manifest同时绑定raw manifest SHA和valid receipt SHA。三类mapping receipt及hard-negative receipt随后只绑定valid receipt SHA和各自上游parent/candidate身份，不反向写回raw/valid manifest。
- filtered cache shape唯一固定为masked context=`[N,4,432,768]` FP16、teacher target=`[N,4,768]` FP32，其中`N`分别对应filtered manifest中的`N_train`或`N_eval`固定图像顺序。
- raw与filtered manifest必须绑定repository commit、生成脚本SHA、config SHA、CLIP checkpoint SHA、预处理/层/projection身份、Python/PyTorch/CUDA/cuDNN/GPU、batch16、compute dtype、autocast/TF32/determinism flags、输入图像相对路径顺序SHA，并逐tensor记录shape、dtype和SHA256。
- Gate结果必须报告缓存shape/dtype/字节数、缓存生成墙钟时间和GPU、训练吞吐、单图Top-5验证延迟、峰值显存及相对Parent延迟。
- “隐藏证据”只对predictor隐藏；整个训练系统读取teacher target，论文统一称`predictor-held-out contextualized region-token evidence`。

## 11. 与三创新主线的关系

旧路径：`patch×文本静态相似度 → 聚合/重排/干预 → 修改类别分数`。

候选新路径：`TG构造候选语义 → GTD迁移dev-unseen候选 → 输入前遮挡区域 → HCVC以masked context×候选原型预测teacher region-token residual → 条件能量验证候选`。

若Gate 0、正式GZSL和多seed均成立，HCVC可作为TG与GTD之后的第三核心创新候选：`TG构造假设 → GTD迁移假设 → HCVC以视觉补全验证假设`。后续输出融合默认只是工程接口，不作为第四创新；若需要可学习裁决器，必须另行证明它引入新信号或新表示原语。

module_interface: 输入为同一无泄漏冻结TG+GTD parent的Top-5、768维最终候选原型、4个masked-student visible context和4个teacher target residual；输出为每候选`conditional_completion_energy`和诊断。Gate 0不学习与parent logit的融合权重。

## 12. 当前证据、状态与owner边界

evidence_refs:

- `research/ideas/IDEA-162_learnable_concept_readout_probe.md`
- `research/ideas/IDEA-163_tri_state_evidence_predicate_set.md`
- `research/ideas/IDEA-164_observable_signed_evidence.md`
- `research/ideas/IDEA-165_constrained_evidence_graph_search.md`
- `research/ideas/IDEA-166_text_conditioned_visual_distribution.md`
- `research/ideas/IDEA-167_conditional_information_evidence.md`
- `research/ideas/IDEA-168_concept_specific_region_interaction.md`
- `research/ideas/IDEA-169_contrastive_concept_interaction.md`
- `research/ideas/IDEA-170_content_aware_inpainted_interaction.md`
- `/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-162-learnable-concept-readout-seed7/result.json@sha256:4f73cbbd0308b9e96af1342df2f45bb2f89ed0ffb8ec1bf6001e835101b574af`

performance_status: `proof_of_path_not_run`；不报告U/S/H/ZS，不创建Innovation分支，不进入V4队列。

coverage_and_transfer: 首轮仅CUB、标准100/50开发划分、seed7；跨seed、AWA2、SUN、formal 150-class training及true-unseen official GZSL均未知。单seed/单split不得写“稳定”或“第三核心创新成立”。

downstream_effects: 双Agent定稿与owner的Idea确认已经完成，但本次没有授权实现；owner另行授权实现Gate 0后方可编码。Gate 0通过后才允许建立正式Innovation Experiment，并以同checkpoint验证TG+GTD+HCVC与HCVC-off。

owner_decision: 2026-08-29 owner要求先按仓库规范严谨确定HCVC Idea，并要求两个子Agent直接相互质询直到Idea没有未解决问题；该授权只覆盖Idea定稿和项目规范，不授权实现、创建实验分支、进入队列或启动训练。

## 13. 双Agent对抗定稿记录

review_status: passed_for_proof_of_path_idea_only
review_subject_sha256: `2a6436c6c9699dba2f8c49e9f0eca113d0ea2a40a28cc5551016ebd84fbb9acf`
review_agents: [`/root/xian_agent_a`, `/root/xian_agent_b`]

- 2026-08-30独立首审与直接交叉质询结论：双方共同为`revise`；集中问题为无泄漏TG+GTD父身份、image-only常数错误、普通ranking退化、mask/teacher/student物理边界、实现数值、统计Gate、IDEA-162/170事实边界及最近工作。
- 2026-08-30第一次集中修订复核仍为`revise`；新增关闭项包括fixed-50 parent完整日程、100/50 block-diagonal文本置换、HCVC-Core与HCVC+Rank主次、FP32/FP16缓存身份、有效样本和batch合同、训练期target shuffle及独立训练的泄漏上界。
- 2026-08-30第二次集中修订复核仍为`revise`；新增关闭项包括缓存生成计算身份、不可漂移parent参数、共享初始化与完整AdamW合同、统一`N_train/N_eval`分母、visible-pool退化检查、MSE聚合顺序和精确缓存shape。
- 2026-08-30第三次集中修订复核仍为`revise`；新增关闭项包括parent CUDA确定性入口、三类mapping唯一单循环算法及receipt、固定CLIP checkpoint SHA、无循环cache身份链、稳定候选排序、共享hard-negative receipt和MSE除零失败条件。
- 2026-08-30第四次集中修订复核仍为`revise`；新增关闭项包括HCVC训练/评估独立CUDA确定性入口、100/50分层资产与进程物理隔离、唯一bootstrap矩阵/CI算法及整数exact McNemar实现。
- 2026-08-30第五次集中修订复核仍为`revise`；新增关闭项包括covered排序率的50类macro定义、parent完整150类文本topology轴、train/eval分别无效率、filtered manifest允许键、共享hard-negative的Ranking-only精确loss。
- 2026-08-30第六次集中修订复核仍为`revise`；新增关闭项为150行模型轴与xlsa 200类全局ID之间的唯一global↔local映射、local模型标签、双ID receipt和official文本tensor边界。
- 2026-08-30第七次集中修订复核仍为`revise`；新增关闭项为不连续seen local ID到100列CE position的唯一第三层映射、logit/centroid顺序及hard-negative转换链。
- 2026-08-30最终复核：两名Agent完整读取上述准确`review_subject_sha256`，先独立复核，再直接相互质询；双方均报告`P0=0 / P1=0 / P2=0 / pass`，无分歧。最后关闭的问题是不连续seen local ID到100列CE位置的第三层映射。
- 共同结论：**范式Idea双Agent对抗审核通过（仅授权owner确认后实现Gate 0，不代表Gate成立、Innovation晋级或论文claim成立）。**
- 本次通过后的文件变化只包含`status/review_status`与本审核记录，不改变已审方法、数据、公式、控制或Gate合同；准确审核对象仍由`review_subject_sha256`绑定。
