---
idea_id: IDEA-201
name: Compiled PCLR
short_name: C-PCLR
status: promoted_framework_v7
source_type: owner_hypothesis
problem_category: class_competition
mechanism_tags:
  - pairwise_relation_text
  - graph_supervised_internalization
  - classifier_compilation
  - optional_training_hard_negative_mining
reuse_refs:
  - IDEA-186
evidence_refs:
  - FRAMEWORK-V5-R4 metrics SHA efbdca19f8248b2e16c99baa7aa5a81d2279218db910a9a00e7303d45d2fc2bc
  - 2026-09-02 read-only all-edge diagnostic recorded in this card
  - 2026-09-02 IDEA-201 Gate A read-only stdout recorded in this card
  - V6-TRY-006 metrics SHA fbbd8ef520d8d6bca62cc1d860a0432a244ab99af30761a3ffd8c824f7c90879
base_commit: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
base_commit_candidate: 52b511d77b4ad048f35b40dc3cbd9afd092167e9
base_ref_candidate: framework/v5 and annotated tag v5^{}
performance_status: above_paper_parent_owner_promoted_v7
test_used_for_selection: true
unseen_images_used_for_gradient: false
strict_blind_claim: false
human_annotations_used: false
expert_attributes_used: false
llm_world_knowledge_used: true
---

# IDEA-201：编译式PCLR（C-PCLR）

> 当前身份：owner已将S/V/I统一方法晋级为`FRAMEWORK-V7`，论文方法父基线固定为TG+GTD。
> 历史`proof_of_path/revise`、程序`drop_gate_b_contract_failed`、低于内部V5/matched online-V5、
> test-selected与未完成成本/多数据集/新颖性对照等事实继续原样保留，不因晋级被删除或改写。

## 1. 唯一问题与核心假设

正式V5的PCLR在每张图上依次执行Parent Top-17、438边方向打分、动态mask、Laplacian
映射、逐图cap/std调制、late role ensemble和seen gamma。它能达到
`U/S/H/ZS=80.694097/81.446952/81.068777/88.785273`，但部署出口仍是一条逐图关系
求解链，训练阶段的关系知识没有形成可直接使用的冻结分类器。

**问题**：能否让关系图只在训练和导出阶段提供结构监督，把它内化为图像Reader与类别权重，
从而在不读取关系边、不求解Laplacian、不做动态Top-K的部署前向中保留相对准确父条件的真实
H提升？

**可证伪假设**：固定关系方向矩阵和Laplacian映射可先编译为关系类别原型；若Reader、关系
强度和角色融合再接受最终200类分类CE，那么关系证据可以被内化到一个冻结的`Q,b`分类出口。
真实部署只执行`h(x)Q^T+b`，同时Full高于准确Parent，且S/V/I三个预注册关闭各自造成至少
`1.0 H`下降。训练期Top-17若启用，只能作为不反传的困难负类采样器；最小首跑优先不启用
Top-17，以免把旧动态局部选择机制混入新路径。

**唯一核心改动**：把V5的逐查询关系图求解改为训练/导出期的关系图原型编译与端到端内化；
不是继续调Top-K、ridge、cap、scale、role或gamma。

## 2. 准确数学定义

冻结438条无向边及每条边的两个方向文本向量：

```text
D[e] = text(A rather than B) - text(B rather than A)   # [438,768]
B[e,a] = +1, B[e,b] = -1                              # [438,200]
M = (B^T B + lambda I)^-1 B^T                         # [200,438]
G = M D                                                 # [200,768]
```

对不含逐图mask、逐图cap和逐图std的线性关系部分：

```text
u D^T M^T = u G^T
```

其中`u=Reader(x)`。该恒等式只允许支持“无mask线性关系部分可以编译”，不能声称完整R4
与编译模型代数等价。若旧PCLR对节点势能做逐图类别轴中心化，该中心化在200类联合
softmax/argmax下只增加一个逐图常数；编译候选可用softmax平移不变性忽略它，但必须在
logit等价测试里单独记录是否包含该常数。

训练/导出候选定义：

```text
u(x) = normalize(x + W2 GELU(W1 x))
relation_logits = alpha * u(x) G^T
image_logits = x Q_image^T
full_logits = image_logits + relation_logits + b
h(x) = concat(x, u(x))
Q = concat(Q_image, alpha * G)
deploy_logits = h(x) Q^T + b
```

`Q_image`由训练后的基础TG/GTD类别权重和固定角色原型融合导出；角色项必须是`x`到类别logit
的固定线性原型项，融合权重必须是图像无关常数或固定每类参数。若角色分支存在各自logit
归一化、图像依赖权重或融合后归一化，就不能写入`Q_image`，也不能声称部署为单一`hQ^T+b`。
R4的seen gamma在现有代码中是对seen列减去固定常数，因此只能作为加性类别bias写入
`b_seen -= gamma`；若后续实现把gamma改成乘法、温度或输入依赖校准，则不能再声称它被bias
吸收。八角色不是部署时late ensemble。`Q`必须显式导出为`[200,1536]`，`b`为`[200]`。
`alpha`采用有界参数化，并结合固定`G`行范数与归一化`u`约束关系残差幅度；部署路径不得保留
逐图class-axis cap、parent-logit std调制、late role或late gamma分支。

训练只使用official seen图像和seen标签。全部200类冻结文本类别/关系资产可作为语义负类，
但unseen图像及其标签不得进入梯度。冻结资产包括类别轴、`B/D/M/G`、CLIP文本向量和
IDEA-186关系句；训练参数必须在实现前列明，至少区分TG/GTD基础类别权重、Reader
`W1/W2`、`alpha`、角色融合权重和导出bias。最终200类分类CE必须直接更新预注册的可训练
参数；方向关系CE保留为Reader的辅助监督，但只允许给包含真类的边打方向标签。对样本类别
`y`，定义`E_y={e | y=a_e or y=b_e}`，方向目标由`B[e,y]`符号决定；两个端点都不是`y`的边
必须ignore，不能强行标注为某个方向。

最小首跑固定禁用Top-17。只有后续得到owner新的预注册授权时，Top-17才可作为可选训练期困难
负类或困难关系边选择，并与`E_y`求交后进入方向CE；交集为空
时该样本跳过方向CE。hard-negative manifest必须只由seen训练图像forward、seen标签、冻结200类
文本/关系资产和预训练/训练中父模型输出产生；mining不反传；不得读取official unseen图像、
unseen标签、official test confusion、test错误统计或看过test后的规则来生成训练样本。Top-17
不得改变部署forward，也不得单独包装成模块贡献。

## 3. 旧路径、新路径与非等价边界

### old_solution_path

```text
image x
→ TG/GTD Parent logits
→ per-image Parent Top-17
→ 438 pairwise direction scores
→ both-endpoint dynamic edge mask
→ regularized Laplacian solve
→ per-image center/cap/std correction
→ late role0/role6 ensemble
→ late seen gamma
→ 200-class logits
```

### new_solution_path

```text
train/export: frozen pairwise graph (B,D) → G=MD → jointly learned Reader/alpha/roles → Q,b
deploy: image x → h(x)=[x,Reader(x)] → h(x)Q^T+b → 200-class logits
```

### principle_difference

旧路径把关系图当作每个查询都要运行的在线求解器；新路径把关系图当作训练期结构监督和类别
权重生成约束，部署时只保留被内化的视觉读取函数与冻结类别矩阵。它改变关系推理发生的阶段
和状态：从逐图边状态/节点势能，改为一次性关系类别原型和学习后的分类器参数。

### non_equivalence_test

1. 完整R4含输入依赖Top-17 mask、class-axis cap和parent std，不能写成一个固定`Q,b`；改变
   某张图的Top-17集合会改变激活边集合，这是固定线性类别矩阵无法代数精确复现的反例。
2. 编译候选必须真实删除部署路径的candidate mask、edge tensor输出、Laplacian solve、cap、
   parent std、late roles和late gamma；只改函数名或缓存`M`不算新路径。
3. 导出的`hQ^T+b`必须与导出前的图无关候选forward在真实batch上逐logit一致，最大绝对误差
   不超过`1e-5`，且该验证必须覆盖完整200类logits而不只是关系支路。若仍需关系图对象才能
   得到同一输出，非等价主张失败。
4. 新颖性不可还原必须作为独立P1关闭。最小对照为：`G=MD`对比GCN/DGP-style可学习
   graph-to-classifier；`G=MD`对比degree-normalized `B^T D`局部聚合、simple average/nearest
   comparative descriptor prototype或PC-CLIP式类原型；Reader+最终CE内化对比frozen semantic
   prototype only和同参数量无图残差head；真实`G`对比保度边-文本置乱`G_shuffle`与同谱/同行范数
   随机`G_rand`。任一对照同预算同协议匹配Full，则C-PCLR只能降级为已有graph classifier
   synthesis与comparative descriptors的实现组合。

## 4. 已有直接证据与当前优势

### 正式父结果

- TG+GTD准确Parent：`H=79.070015`。
- V5 R4：`U/S/H/ZS=80.694097/81.446952/81.068777/88.785273`。

### 2026-09-02只读all-edge诊断

固定R2 checkpoint和全部R4条件，只把Top-17动态mask改为全438边：

| 条件 | U | S | H | ZS |
|---|---:|---:|---:|---:|
| R4 Top-17 | 80.694097 | 81.446952 | 81.068777 | 88.785273 |
| R4 all edges | 80.754775 | 80.373096 | 80.563484 | 88.468552 |

`Delta H=-0.505293`，但all-edge仍比Parent高`+1.493469 H`。Top-17→all-edge的GZSL
纠正/破坏为`47/66`，净`-19`；ZS纠正/破坏为`19/28`，净`-9`。Top-17 R4逐指标
复现误差为0；R3 all-edge H复现账本误差为`2.3e-7`。

这只证明动态mask不是全部性能的必要条件，并给编译方向提供风险上限代理；它没有执行新的
端到端训练，也没有删除cap/std/late roles/gamma，因此不构成编译模型的准确率或速度优势。

### current_advantage

- `accuracy`风险/可行性代理：旧checkpoint的all-edge条件仍比准确Parent高`+1.493469 H`；
  这不是C-PCLR真实accuracy advantage。
- `speed_or_cost`：数学上可把438边线性映射折叠为200个关系原型，但尚无真实部署计时。
- `generality`：未验证。

因此当前只能标记`proof_of_path`，不能登记为`innovation`或`paper_core_innovation`。all-edge
诊断只作为feasibility motivation，不作为IDEA-201性能证据。

### 2026-09-02 Gate A只读结果

owner确认准确父commit为`52b511d77b4ad048f35b40dc3cbd9afd092167e9`。服务器当前代码中
PCLR/GTD/TG与评估相关文件相对该commit无diff；固定R2 checkpoint代码身份为
`b0a756dd624e883eb50d19a2455ba06bdc73f118`、config SHA为
`0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49`。

使用50张seen训练图、50张official test-seen图和50张official test-unseen图，只在内存构造
`M/G/Q/b`，不写结果文件：

| 检查 | 结果 | 门槛 |
|---|---:|---:|
| `uD^TM^T` vs `uG^T`最大绝对误差 | `4.218847e-15` | `<=1e-5` |
| 中心化后最大绝对误差 | `4.218847e-15` | `<=1e-5` |
| 同时翻转`B_e,D_e`后的`G`误差 | `0` | `<=1e-5` |
| standalone Reader vs model Reader | `0` | `<=1e-5` |
| 固定仿射候选pre-export vs `hQ^T+b` | `5.722046e-6` | `<=1e-5` |

导出`Q`形状为`[200,1536]`，`b`为`[200]`，输出全部有限；导出后前向只使用Reader权重、
`Q`和`b`，不读取在线关系图。438边图有3个连通分量、最大分量192类、无孤立节点，度数
范围`3..12`，`B^TB+0.3I`条件数`45.334843`；`G`行范数范围
`0.198690..0.534117`。Gate A判定为`pass`。

该通过只证明无mask线性关系和固定仿射分支可被数值稳定地编译/导出。Gate A使用的是删除R4
逐图role标准化、std和cap后的结构代理，不是正式C-PCLR checkpoint，不复现R4 logits，也没有
产生U/S/H/ZS或真实速度优势；`performance_status`继续保持`proof_of_path`。

### 2026-09-02 Gate B fixed-200正式结果

`V6-TRY-006`在准确父commit独立分支完成seed7、batch50、200名义epoch、28,228 updates和
202行完整official评估历史。TG/GTD、matched online-V5 Reader/beta与C-PCLR头使用同一父轨迹
和同一训练预算；C-PCLR固定禁用Top-17。best-Full-H checkpoint=`update13818`：

| 条件 | U | S | H | ZS | Full−off H |
|---|---:|---:|---:|---:|---:|
| C-PCLR Full | 77.606910 | 83.639657 | 80.510432 | 88.473403 | — |
| S-off | 76.141131 | 82.428479 | 79.160157 | 87.064338 | 1.350275 |
| V-off | 82.206428 | 76.821315 | 79.422694 | 88.181764 | 1.087737 |
| I-off | 82.451552 | 76.188660 | 79.196481 | 88.189560 | 1.313951 |

matched online-V5同预算best为`U/S/H/ZS=80.112976/81.535739/80.818096/88.646406`
`@ update13818`；正式V5为`H=81.068777`。C-PCLR相对两者分别为`-0.307664 H`与
`-0.558345 H`。S/V/I三个部署依赖均超过预注册`1.0 H`门，但Full没有超过任一父条件；程序
正式decision=`drop_gate_b_contract_failed`，Idea状态改为`rejected_gate_b_below_parent`。

结果证明关系编译路径可运行且三个组成部分均产生非平凡依赖，但没有形成当前准确率优势，不能
登记为`innovation`或`paper_core_innovation`，不触发Gate C或参数补救。输出URI：
`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/compiled_pclr/V6-TRY-006`。

## 5. 最近工作与允许的新颖性边界

下列原始论文页面于2026-09-02重新核对；当前只作Idea检索证据，尚未建立正式PAPER卡或
本地PDF/SHA：

1. Wang, Ye, Gupta, **Zero-Shot Recognition via Semantic Embeddings and Knowledge Graphs**,
   CVPR 2018：类别图和语义向量生成视觉分类器权重，推理使用生成后的分类器。
   https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Zero-Shot_Recognition_via_CVPR_2018_paper.html
2. Kampffmeyer et al., **Rethinking Knowledge Graph Propagation for Zero-Shot Learning**,
   CVPR 2019：DGP从类别知识图预测seen/unseen分类器权重并用于简单分类。
   https://openaccess.thecvf.com/content_CVPR_2019/html/Kampffmeyer_Rethinking_Knowledge_Graph_Propagation_for_Zero-Shot_Learning_CVPR_2019_paper.html
3. Sam et al., **Finetuning CLIP to Reason about Pairwise Differences**, arXiv 2024、2025修订：
   用LLM合成差异描述训练CLIP，并提出comparative prompting分类。
   https://arxiv.org/abs/2409.09721
4. Lee et al., **Enhancing Visual Classification using Comparative Descriptors**, WACV 2025：
   为相似类别生成comparative descriptors并在CLIP分类中筛选/融合。
   https://arxiv.org/abs/2411.05357
5. Duan et al., **Visual-Semantic Graph Matching Net for Zero-Shot Learning**, TNNLS 2024：
   使用视觉/语义图和跨图关系约束训练ZSL表示。
   https://arxiv.org/abs/2411.11351

**禁止claim**：图生成分类器、训练时用图而推理时线性分类、比较描述、类别关系监督、
Laplacian传播中的任一项是本项目首次提出。

**仅允许继续审查的窄claim**：在CLIP-GZSL中，把冻结的成对形态差异文本方向先经固定
Laplacian逆编译成关系类别原型，并用共享图像Reader、方向CE和最终分类CE把该关系监督内化，
最终将基础/角色/关系出口统一导出为不含在线图推理的`hQ^T+b`。该组合是否仍只是GCN/DGP、
comparative descriptors和标准classifier synthesis的直接拼接，是本次双Agent必须证伪的核心。

### closest_paradigm_work

最接近的两条边界分别是DGP/GCN-ZSL的“类别图→分类器权重”和Pairwise Differences/
Comparative Descriptors的“差异文本→分类”。C-PCLR只有在证明其固定边方向到节点原型的
解析编译、图像条件Reader监督、最终CE内化及完整无图部署构成不可由上述方法直接替换的统一
训练对象时，才有窄而真实的方法级新意；不主张范式级或首次。具体地，不能把“图生成分类器”、
“比较文本改善分类”或“最终线性head”写成贡献；只允许审查“有向比较文本边场经正则
incidence/Laplacian积分成为类残差原型，并被严格编译到部署分类头”这一窄组合。

## 6. 最小可行、最小证伪与实验合同

### minimal_viability

1. 在真实CUB batch上，无mask线性关系分支的edge-space与`G=MD` prototype-space logits
   最大绝对误差`<=1e-5`；同时随机翻转若干边的`B_e,D_e`符号，重新求解的`G`应保持不变。
2. 导出前图无关Full forward与导出后`hQ^T+b`逐logit最大绝对误差`<=1e-5`；验证不得包含逐行
   `G`归一化、`concat(x,u)`后整体归一化、图像依赖gate/temperature/cap/std或late ensemble。
3. 部署代码路径不读取`edge_index/relation_embeddings/incidence/laplacian_map`，不执行Top-K、
   solve、cap、parent std或late ensemble。
4. 一次真实端到端RUN产生有限U/S/H/ZS，并使用同一best-H checkpoint报告。

### minimal_falsification

先只做两个Gate，不提前建设完整框架：

- Gate A：固定真实batch验证`uD^TM^T == uG^T`、边方向不变性和最终导出等价；任一误差超门或
  仍需图对象，立即否定“可编译出口”。`M`实现必须使用线性方程求解或Cholesky，不直接显式求逆；
  固定`lambda`、dtype和容差，并报告连通分量、度分布、孤立点、矩阵条件数与`G`行范数。
- Gate B：在owner确认父commit后实现最小联合训练，只允许一个完整训练条件；若Full不高于准确
  Parent，或任一S/V/I关闭贡献不足`1.0 H`，立即drop，不搜索Top-K/ridge/cap/scale/gamma补救。
  单RUN只能用于drop或初步keep；若要升级为论文候选，需至少3 seed或owner明确接受的固定复算
  协议，并报告均值/方差、逐类配对或分层bootstrap、best-H同checkpoint U/S/H/ZS和独立best-ZS。
- Gate C：若Gate B初步keep，再运行三组不可还原对照：GCN/DGP-style graph-to-classifier、
  simple/nearest comparative descriptor prototype、frozen semantic prototype only。任一对照匹配，
  立即撤销窄新意claim。

### Full与关闭路径

同一个Full checkpoint，不重训：

- `S-off`：从`Q_image`中严格减去角色原型线性项，保留基础原型、Reader和编译关系残差。
- `V-off`：关闭learned relation-reader残差，使用同形状冻结基线`u0=normalize(x)`，保留关系
  原型、`alpha`和其他接口；这验证Reader学习贡献，不等于关闭整个关系残差。若owner要求
  `relation_logits=0`作为V-off，该定义会与I-off重复，必须在运行前重新定义模块合同。
- `I-off`：`alpha=0`，关闭整个编译关系残差；Top-17不是独立模块关闭。

`Reader-nonlinearity-off`可作为机制消融名称保留，但不能替代上述V-off来证明项目合同中的
V模块贡献。所有off均使用同一个Full checkpoint，不重训，并记录off后张量形状、归一化和
forward接口。

三项均要求`H_full-H_off >= 1.0pp`，这是owner当前性能合同，不是新颖性证明。三项同checkpoint
关闭只能称为deployment dependency knockout，不能声称重训后不可替代；若论文要主张S/V各自独立
贡献，还需要等预算重训消融。Full还必须在相同数据、骨干、训练预算、checkpoint选择和评估协议
下高于owner确认的准确Parent；`H=80`是目标而非硬通过线。准确Parent、source checkpoint、split、
关系资产、角色原型、类别文本、`lambda`、边顺序、方向约定、`G/Q/b`哈希和evaluator必须由owner
确认并固定。

### 最小成本证据

只测同硬件、同dtype、batch=1和正式评测batch、同预载特征的head-only参数量、模型文件大小、
峰值显存、p50/p95延迟和吞吐：Parent、R4在线关系头、导出`hQ^T+b`三者同时报告。不把CLIP特征
提取时间混入关系头claim。没有真实计时前不得声称更快；若导出头只快于R4但明显慢于Parent，
结论必须限定为“移除逐图图推理”，不能声称接近Parent成本。

## 7. 数据、选择与失败边界

- 继续采用`chen_shiming_code_aligned_test_selected_gzsl`；允许official test选择整模型
  checkpoint/最终配置，必须披露`test_used_for_selection=true`、
  `nested_official_test_selection=true`、`strict_blind_claim=false`。该结果只能作为Chen-style
  test-selected描述性结论；若要对外主张泛化优势，必须在未参与选择的划分或数据集上确认，并披露
  候选数、选择阶段和所有看过official test的超参数。
- `unseen_images_used_for_gradient=false`；不得从official unseen图像、标签或错误统计生成
  hard negatives、关系文本、角色权重或训练样本。
- 关系资产沿用IDEA-186冻结的438边/876句/`[438,2,768]`身份；资产变化属于另一Idea。
- `failure_boundary`：可能损失输入依赖Top-17带来的局部去噪；固定`G`会把无关远边噪声带入
  类别原型；Reader可能退化为身份映射；角色或alpha可能吸收全部提升，使V/S/I任一贡献不足
  1点；seen-only训练可能无法把最终CE收益可靠迁移到unseen；简单出口也可能只是旧GCN/DGP
  classifier synthesis的特例；若三组不可还原对照任一组匹配Full，则核心新颖性claim必须
  降级或撤销。

## 8. 论文级claim上限

### paradigm_shift

关系图从“每张查询图像都运行的在线求解状态”变为“训练/导出期生成并约束冻结分类器的监督
对象”。当前只作为待证伪的新求解路径描述，不声称范式级创新。

### why_not_module

它不是只增加Gate、Head或校准项：验收要求完整删除在线Top-K/edge/Laplacian/cap/std/late
ensemble链并导出统一分类器。然而，它仍可能被最近工作证明只是graph-to-classifier synthesis
与comparative descriptors的直接组合；在双Agent和真实对照关闭前不得称创新成立。

### paper_level_claim

若全部合同成立，只允许声称：一种面向细粒度CLIP-GZSL的关系图监督内化方法，将冻结成对有向
差异文本经正则图积分解析编译为关系类别残差原型，并把基础、角色和关系证据统一导出为无在线
图推理的简单分类出口。禁止“首次”“新范式”“图生成分类器首次”“比较描述首次”“精确复现动态
Top-17”或“训练一定追回mask增益”。

## 9. 双Agent对抗记录

- 审核日期：2026-09-02
- 第一轮原草稿内容SHA256：`e70c82790fb706010383716eaf9633331f66348b673e823ad43863a6951e8adb`
  （由主Agent在两名Reviewer开始独立检查前计算）。
- Agent A第一轮独立结论：`revise`，`P0=0/P1=5/P2=3`。核心P1覆盖父commit未确认、
  proof-of-path证据边界、S/V/I关闭口径、训练/冻结/导出参数边界和hard-negative数据边界。
- Agent B第一轮独立结论：`revise`，`P0=0/P1=7/P2=4`。核心P1覆盖新颖性不可还原、
  完整`Q,b`导出等价、hard-negative数据边界、统计门、复杂度对照和V-off定义。
- 审核异常：两名Reviewer均违反只读约束并发修改同一草稿，且返回了互相矛盾的第二轮状态；
  Reviewer A报告交叉质询未有效完成并保持`revise`，Reviewer B却报告已经`pass`。因此所有由
  Reviewer自行写入的第二轮SHA、交叉记录和通过语句全部作废，不能作为双Agent签字。
- 主Agent集中修订：保留两份第一轮清单中可验证且不冲突的修订，包括合法incident-edge方向
  CE、训练期hard-negative边界、`Q,b`/gamma/role可导出条件、V-off定义、softmax中心化、
  GCN/DGP与comparative-descriptor不可还原对照、统计门、Parent身份固定要求和成本claim边界；
  同时删除矛盾的通过结论。后续复审只读取下方锁定payload，不允许Reviewer写文件。
- 两名Reviewer只读复审payload SHA256：`49571f561816b6c9c1f4b56a88a31daf6743a35bbf9a91eecc874bc358360ef0`
  （计算本文件从开头到`## 9. 双Agent对抗记录`标题之前的UTF-8内容，包含标题前换行字节；
  审核记录自身不参与hash）。
- Agent A只读复审：`revise`，`P0=0/P1=2/P2=2`。P1为准确父commit未获owner确认、
  C-PCLR尚无真实current advantage；P2为hard-negative标签名称和solve/Cholesky实现提示。
- Agent B2只读复审：`revise`，`P0=0/P1=7/P2=3`。除同意A的两个P1外，还要求在Gate中
  关闭GCN/DGP/classifier-synthesis还原反例、固定完整`Q_image/Q/b`导出公式与参数边界、
  首跑禁用Top-17、预注册统计匹配阈值和真实成本对照。
- Gate A前主Agent修订payload SHA256：`5e718fc5e1b791627dbd18050ae3b85e4d47fca19924beea2970805279103db0`。
  相对复审payload只吸收A的P2命名修正并把最小首跑Top-17从“可选”收紧为“固定禁用”；
  该版本没有获得新的Agent pass，继续保持`revise`。
- 最强共同反例：若同预算DGP/GCN-style graph-to-classifier、simple comparative descriptor
  prototype、无图同参数残差head或`G_shuffle/G_rand`任一条件匹配Full，则C-PCLR只能降级为
  已有graph classifier synthesis与comparative descriptors的组合实现。
- 有效直接交叉质询：未完成。两名临时子Agent均明确报告当前运行环境没有
  `collaboration.send_message`或等价的子Agent互传工具，不能用主Agent转述冒充直接质询。
- 最终结论：`proof_of_path / revise`，不是“双Agent对抗审核通过”。本轮不再增加Agent或伪造
  签字；在Gate B/C关闭剩余P1前，不授权创建创新分支、进入队列、登记
  `innovation`或启动正式RUN。
- Owner父条件确认：2026-09-02确认`framework/v5` commit
  `52b511d77b4ad048f35b40dc3cbd9afd092167e9`为准确父commit；对应P1已关闭。
- Gate A：2026-09-02只读结构诊断通过；公式、方向翻转、Reader和`hQ^T+b`导出误差均低于
  `1e-5`。它关闭“数学上不能编译/导出”的风险，但不关闭current advantage、新颖性对照、
  统计门或S/V/I真实贡献P1，因此Idea仍为`proof_of_path / revise`。
- Gate A回填后payload SHA256：`96704bd3a6424aa1be8c28463df51bb485d12b2d9f19b5b068fa824cb17ece1f`。
- Gate B最终覆盖：fixed-200正式RUN完成并判定`drop_gate_b_contract_failed`；`performance_status`
  从`proof_of_path`更新为`below_parent`。该结果关闭本Idea，不继续Gate C或补救搜索。
- Owner覆盖决定：owner于后续对话明确“这次就算通过”。该决定把IDEA-201保留为效率型论文候选，
  不改写程序原始`drop_gate_b_contract_failed`与真实数值，不等于准确率门通过，也不自动晋级
  `innovation/framework`。后续若用于论文，必须补真实延迟/显存/吞吐、多数据集与最近工作对照，
  claim限定为精度—部署成本折中和窄方法组合。
- FRAMEWORK-V7晋级：owner进一步明确论文不以内部V5为父框架，采用TG+GTD作为公开方法父基线，
  并将S/V/I统一方法晋级为首个正式论文框架`FRAMEWORK-V7`。相对TG+GTD `H=79.070015`，
  V7 Full提升`+1.440417 H`；内部V5和matched online-V5结果继续保留为开发事实，不删除、不改写。
