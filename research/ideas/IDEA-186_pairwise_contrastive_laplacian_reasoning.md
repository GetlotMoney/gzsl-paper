---
idea_id: IDEA-186
name: Pairwise Contrastive Laplacian Reasoning
short_name: PCLR
status: keep_r4_H81_gate_passed
base_commit: f87d1af87c3b56d04dadd46c91dcf1ed57309d25
parent_run: TUNE-002-RUN-030
parent_H: 79.070015
experiment: V4-TRY-023
human_annotations_used: false
expert_attributes_used: false
llm_world_knowledge_used: true
test_used_for_selection: true
unseen_images_used_for_gradient: false
---

# IDEA-186：Pairwise Contrastive Laplacian Reasoning（PCLR）

## 唯一研究问题

TG+GTD把类别分别建成点原型，但CUB错误通常发生在视觉相近类别之间。单类别描述会
重复“small bird、brown wings”等共有内容，模型缺少“相对哪个近邻，什么差异才重要”
的监督。PCLR不再增加一个独立类别原型，而把类别关系图上的有向差异句当作边证据，
再用固定Laplacian逆问题把边证据还原为全部200类的相对支持势能。

## 准确父条件和资产边界

- Parent：`TUNE-002-RUN-030`，`U/S/H/ZS=76.164645/82.205832/79.070015/86.955839`，
  best update=`14241`。
- 视觉、八角色文本和split只绑定Parent dynamic-v3 manifest SHA
  `3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6`。
- 图只由200个冻结display class name的官方OpenAI CLIP类名向量构造：模板
  `a photo of a {class}`，每类top-3，取无向union后固定438边；seen诱导图最小度为1。
- 每条边生成两个方向句：`A rather than B`与`B rather than A`，共876句。只允许
  可见形态，不允许栖息地、行为、地理、专家属性、部位标注、框或分割。
- 明确披露`llm_world_knowledge_used=true`与`human_annotations_used=false`。关系句是
  冻结外部语义，official test图像和标签不参与图、句子或CLIP编码。

## 固定Full方法

共享reader为`z=normalize(x+W2 GELU(W1 x))`，hidden=`64`。`W1`使用独立seed
`18601` Xavier初始化、bias为0；`W2`权重和bias全0，因此初始化严格等于冻结CLS。
对seen类训练图像，仅在两个端点均为seen且与真类相连的边上做双向关系二分类，按图
平均incident-edge loss，避免类别度数改变样本权重。

固定有向incidence矩阵`B[e,a]=+1,B[e,b]=-1`。每图边差`r_e=s(a>b)-s(b>a)`，
以`M=(B^T B+I)^-1 B^T`得到节点势能`d0=M r`；逐图中心化后固定缩放为
`d=0.5*d0/max(||d0||∞,0.5)`。最终
`logit_full=logit_TG+GTD + beta*stopgrad(std(logit_TG+GTD))*d`，其中
`beta=0.25*sigmoid(rho)`且初值0.05。std始终在完整200类轴计算，ZS只在最后截取
unseen列，保证同一图在GZSL/ZS下使用同一校正。

Parent严格保留RUN-030原CE、topology和GTD gate loss。reader只接收关系loss；beta
只接收`CE(stopgrad(parent logits)+beta*stopgrad(std*d), seen_label)`。辅助Adam复用
gate的`1e-4→1e-5` warmup/cosine，weight decay为0。所有条件固定150 nominal epoch、
21171 updates、batch50、每141步official test评估并选择整模型最高H。

## 成立、失败与关闭条件

通过需best `H>=80.070015`，且相对RUN-030和同checkpoint PCLR-Off均至少`+1.0 H`；
`|U-S|<8`、ZS下降不超过0.5、net correction至少20。PCLR-Off完整历史必须逐更新
复现RUN-030，尤其`H=79.070015 @ update=14241`；否则是工程身份失败，不得解释为
方法涨跌。

Full未过门只允许预注册一次无重训边界救援：同checkpoint把potential cap从0.5改为
1.0，且仅当`0<delta_H<1`、方向诊断成立、ZS与gap安全时触发。mapping shuffle、
generic difference和NoProjection仅在Full过门后运行。此前成对logit选择、角色差值、
图高通和prototype transport都不能直接提供“冻结有向差异文本→共享视觉读取→全图一致
势能”这一闭环，但pairwise description、图Laplacian和差异描述均有先例；正式论文
claim前仍须重新核对最近工作，当前不得称范式级或首次。

## Idea双Agent准入结论

本Idea已在实现前完成两名Agent独立质疑与一次直接交叉：确认它不是既有Top-K重排、
pair selector、角色残差或GTD prototype transport的重复；同时把新颖性收窄到上述完整
接口，删除“范式级”表述。结论只允许冻结资产并进入代码实现，不代表代码审核通过、
实验有效或论文新颖性已经成立。新增forward/loss/资产入口仍须对准确最终commit完成
一轮双Agent代码对抗审核。

## 实际结果与最终裁决

- direct-official Full正常完成固定150 nominal epoch/21171 updates；best update=`13818`，
  `U/S/H/ZS=75.926912/82.386708/79.025018/87.248874`。
- 相对准确Parent `ΔH=-0.044997`；同checkpoint PCLR-Off仅`+0.123105 H`；seen与
  unseen合计net correction=`4`。三个独立成功门均失败，最终`drop_no_rerun`。
- 唯一cap救援要求`0<ΔH_parent<1`，实际为负，因此未触发；controls同样不运行。
- Parent/GTD最终34个张量、optimizer、scheduler与RUN-030 bitwise一致，全部152点Off
  U/S/H一致；但update=`18471`的Off-ZS因PCLR关闭评估少一次prototype重归一化而为
  `86.734289`，RUN-030为`86.774284`，其余151点四指标一致。
- 两名Reviewer共同裁定RUN身份为`engineering_failed_reporting_deviation`：不可作为论文
  canonical数值或完整Off轨迹复现；但该偏差不影响Full H、H选点或三个失败门，足以停止
  投入并淘汰方法，无研究价值再跑150轮。
- 防复发：以后任何模块Off评估直接复用现有`evaluate/_predict`的prototype normalize
  语义，并先做seen/unseen/ZS逐预测parity回归；不能只检查U/S/H。

## Owner重新授权的唯一局部救援

Owner于2026-08-31重新授权一次机制级救援，不改关系文本、reader、Laplacian资产或
Parent训练协议。只读诊断先否定了seen/unseen分别中心化（最高`H=79.089677`）和锁定
Parent组判断（最高`H=79.158583`）；随后确认全438边对每图传播无关关系噪声。

唯一救援固定为：在完整200类Parent logits上取停止梯度的Top-20，仅保留两个端点都在
Top-20内的关系边，其余边差置零后继续使用原固定Laplacian map；最终校正固定乘
`1.25`。旧checkpoint只读诊断得到`U/S/H/ZS=75.991195/83.133650/79.402125/
87.683898`，相对RUN-030为`+0.332110 H`、相对同checkpoint Off为`+0.500212 H`，
仍距`80.070015`差`0.667890`。该数字使用official test选择Top-K和scale，只作为启动
一次救援的乐观诊断，不能作为正式结果。

初始冻结身份`d5f59aa2dd60ff903dd0f84bedc887be046d09b5`在双Agent交叉中发现父轨迹
过强声明P1；集中修复后的救援代码身份为`9028fd79c415f3cac670b1644d77403920b1f4e7`，
config SHA为`606b0e4d3b69cb3b750d275e13b960bca26025ba72d1a6948964f099b5dd7093`。新增forward和
beta loss语义必须先完成同一冻结身份的双Agent对抗审核、真实GPU micro和Off完整parity，
通过后才允许唯一direct-official 150轮Full；未过原`+1 H`门则永久关闭PCLR家族。

正式RUN签字身份固定为：运行代码commit
`9028fd79c415f3cac670b1644d77403920b1f4e7` + 审查声明提交tree
`4d275c74ce8f990170546a8dbfa5d15fdfbc10e5` + config SHA
`606b0e4d3b69cb3b750d275e13b960bca26025ba72d1a6948964f099b5dd7093` + relation
manifest SHA `0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4` +
environment/GPU fingerprint SHA
`8b3e2d5d93cdd9763843c3c5f72903f466a86f7524c9dc2b02bb1d4699c32c59`。Parent完整轨迹
另固定SHA `10591bb35a51949a1989ae3a918b50bca37c1f465a52c6bb5df5552c1b0a4779`。

## Local-PCLR正式结果与家族关闭

唯一direct-official fixed-150救援正常完成`21171` updates、`152`个评估点和`150`次
teacher refresh。best update=`13818`：

`U/S/H/ZS=75.991195/83.094436/79.384234/87.683898`。

- 相对RUN-030 Parent `ΔH=+0.314219`；距离`H=80.070015`仍差`0.685781`。
- 同checkpoint Local-Off为`U/S/H/ZS=76.212001/81.788653/78.901913/87.016541`，
  Local-PCLR为`+0.482322 H`；相对原PCLR `79.025018`提高`+0.359216 H`。
- `|U-S|=7.103240`与ZS安全门通过；seen/unseen净纠错分别`+23/-6`，合计`17`，低于
  预注册`20`；Parent增益、同checkpoint增益和净纠错三项核心门均未通过。
- best-ZS独立观察为`87.935549 @ update=18048`，不与best-H拼接。
- RUN-030的152点评估历史已按update逐点比较，Off的`U/S/H/ZS`全部一致；
  `module_off_full_history_reproduced=true`，旧PCLR报告偏差已关闭。
- metrics/model/evaluation history SHA依次为
  `4be533a63fc25a11bfe0cd09ad9797da7e17215355f7e55e40fec5b8bfc31a21`、
  `3fe687a08a55e29618efa895691075c8afba239c7b42d01e434d780df189174f`、
  `43a3e54ecdf95f1722461f7bf96463d78defbdcda1ff1c217cb10f306d9b15a8`。

最终裁决：局部Top-20关系边确实比全438边更有效，且正式结果接近只读诊断`79.402`，
但增益仍主要来自S上升并伴随U下降，达不到论文核心创新门。IDEA-186及PCLR家族永久
`drop_final_rescue_gate_failed`；不再运行参数补救、controls或新的PCLR变体。

## Owner覆盖关闭结论：唯一R2联合调参

Owner明确认为`+0.314 H`属于难得的真实提升并要求继续。对R1正式best checkpoint只读
扫描后，Top-15、ridge=`0.03`、correction scale=`2.38`、seen-logit gamma=`0.525`
得到`U/S/H/ZS=79.565275/80.288148/79.925077/87.938917`，距`80.070015`仅
`0.144938`。同checkpoint只做最优gamma的Parent最高`H=79.027716`；固定R2 gamma下
calibrated Off为`H=78.539576`，R2 Full相对它`+1.385500 H`，因此不是纯校准假增益。

Owner只授权这一组进入最终R2，不再搜索更多网格。R2代码commit为
`b0a756dd624e883eb50d19a2455ba06bdc73f118`，config SHA为
`0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49`。gamma只在
official Full/Calibrated-Off评估中应用，不进入seen-only beta loss；Raw Off继续与
RUN-030的152点逐点硬校验。R2仍须同时达到原`H>=80.070015`、相对Parent和Raw Off
至少`+1 H`、gap/ZS/net门，否则最终关闭。

R2正式RUN签字身份：代码`b0a756dd624e883eb50d19a2455ba06bdc73f118` + 审查声明tree
`3f07990b8f9c86f543fe4beebb1693f7107b0cf6` + config SHA
`0861877ae3e4725e29aff547d45e0b6d56a186179309acb5493c5906b803fd49` + relation
manifest SHA `0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4` +
environment/GPU fingerprint SHA
`8b3e2d5d93cdd9763843c3c5f72903f466a86f7524c9dc2b02bb1d4699c32c59`。

## R3正式结果

正式只读复算与双GPU micro逐字节一致，metrics SHA为
`39bea2dbf664dc421cd53b2a4f8d219b85f05b9279e7991b596c61d22aa4042a`：

`U/S/H/ZS=77.806163/82.716906/80.186419/87.612945`。

相对RUN-030 Parent、Raw Off、Calibrated Off分别为`+1.116404/+1.284507/+1.692670 H`；
gap=`4.910743`、net=`68`、ZS安全，六项AND门全部通过，正式decision为
`keep_pclr_r3_inference_tune`。必须同时披露effective beta=`0.725859`、上限=`1.7375`、
`nested_official_test_selection=true`和`strict_blind_claim=false`。

## R4：类别角色语义出口

Owner要求双GPU继续搜索到`H>=81`。Reader容量/LR/训练温度、5个Reader初始化及融合、
关系边置信度门控、Top-3/Top-5图融合、coarse patch、中间层特征、train-only类偏置、
pseudo-unseen自适应gamma、独立Parent ensemble和详细文本覆盖均未超过`80.325`。

固定R2 checkpoint与R3关系校正后，额外融合class-level role6（overall appearance）和
role0（beak）logits，权重分别为`0.36/0.16`，seen gamma=`0.91`。诊断得到
`U/S/H/ZS=80.694092/81.446953/81.068771/88.785273`，gap=`0.752861`；U、S和ZS同时
提高，不是单边校准换分。

R4不重新训练或修改source checkpoint；代码commit为
`c80c021e25bdcef5e5a80e2f01286c9f886e3f52`，config SHA为
`f9ec1da6074225F947A2EF0D468E1543445BCC7A6DF6209A181BE025969D98D1`。它是R3之后的
再次official-test超参数选择，必须披露nested selection和nonblind；只有审核后双GPU正式复算
仍`H>=81`且全部门通过，才能完成当前目标。

R4正式签字身份：评估代码`c80c021e25bdcef5e5a80e2f01286c9f886e3f52` + 审查声明tree
`397cbd2eec88b7307a7fcf3eafbf0d9881a57135` + config SHA
`f9ec1da6074225f947a2ef0d468e1543445bcc7a6df6209a181be025969d98d1` + source model
SHA `16b5071f21a3217e58a72315029c28b8cfd97b68f812641bd0145d3f5e0702ab` + 环境fingerprint
`8b3e2d5d93cdd9763843c3c5f72903f466a86f7524c9dc2b02bb1d4699c32c59`。

## R4正式结果

正式结果与双GPU micro逐字节一致，metrics SHA为
`efbdca19f8248b2e16c99baa7aa5a81d2279218db910a9a00e7303d45d2fc2bc`：

`U/S/H/ZS=80.694097/81.446952/81.068777/88.785273`。

- 相对RUN-030 Parent：`+1.998762 H`；相对R3：`+0.882357 H`。
- gap=`0.752854`；Raw seen/unseen净纠错合计`129`；六项AND门全部通过。
- Raw与R3 controls逐`U/S/H/ZS`精确复现source；source checkpoint只读。
- 正式decision=`keep_pclr_r4_semantic_ensemble`，`full_gate_passed=true`。
- 结果必须披露nested official-test selection、nonblind与LLM世界知识使用；它证明当前协议下
  达到81，不支持strict blind泛化声明。

## R2正式结果

R2完整完成fixed-150的`21171` updates、`152`个评估点和`150`次teacher refresh。
best update=`13818`：

`U/S/H/ZS=79.565275/80.288148/79.925077/87.938917`。

- 相对RUN-030 Parent：`+0.855062 H`，距`80.070015`差`0.144938`。
- 同checkpoint Raw Off `H=78.901913`，R2为`+1.023164 H`。
- 同gamma Calibrated Off `H=78.539576`，R2为`+1.385500 H`。
- `|U-S|=0.722873`；seen/unseen净纠错`-24/+100`，合计`+76`；ZS安全门通过。
- 三个辅助门均通过，但总H/Parent增益没有达到预注册`+1 H`，程序正式decision为
  `drop_pclr_full_gate_failed`，不得写成已过门创新。
- 独立best-ZS为`88.112366 @ update=18612`，不与best-H拼接。
- Raw Off 152点`U/S/H/ZS`与RUN-030逐点完全一致，工程身份有效。
- metrics/model/evaluation history SHA依次为
  `3d64bd36e48304b025044b109c579001279400ccec075fc1246496c4f28e8578`、
  `16b5071f21a3217e58a72315029c28b8cfd97b68f812641bd0145d3f5e0702ab`、
  `b7a4e8dc29ee985914bd9c511576db9f2288884045cdb28e8d0f8dd66ab0e910`。

对R2 best再次精扫scale=`2.1..2.7`与gamma=`0.49..0.56`后，最高仍为
`H=79.925072`（scale约`2.38`、gamma约`0.522`），说明当前固定checkpoint已经处于
离散参数平台，继续细调K/ridge/scale/gamma没有证据补足剩余`0.145 H`。

两名Reviewer对正式结果独立复核并直接交叉质询后共同裁定：运行身份和结果合同有效，
`P0=0/P1=0`，但成功门是AND关系；绝对H与Parent增益两项均失败。R2已被明确授权为
唯一最终调参补救，因此不存在未消费的预注册触发条款。任何gamma、scale、Top-K、ridge、
cap、checkpoint或门槛再选择都属于看过official结果后的额外搜索。最终状态为
`drop_final_rescue_gate_failed`，PCLR家族关闭；结果可作为后续独立机制的失败证据，不能
写成已接纳创新。

## Owner再次覆盖关闭结论：R3嵌套推理选择

Owner明确要求继续系统搜索全部超参数，并要求Top-4/Top-5及后续AWA2/SUN测试。该授权
不改变R2已经失败的事实；R3单独标记`nested_official_test_selection=true`且不得称blind。

对R2 best checkpoint联合扫描后，现有正式Top-3图的固定最佳组合为candidate Top-K=`17`、
ridge=`0.3`、inference relation temperature=`0.2`、correction scale=`6.95`、seen
gamma=`0.575`：`U/S/H/ZS=77.806163/82.716906/80.186419/87.612945`。相对RUN-030
Parent、同checkpoint Raw Off和同gamma Calibrated Off分别为
`+1.116404/+1.284507/+1.692670 H`；gap=`4.910743`；seen/unseen净纠错合计`+68`；
固定checkpoint诊断通过全部门。

单gamma及两种无标签双gamma路由均已扫描。最高H自动退化回全局gamma约`0.575`；强制
gap<2时最高仅`H=80.045400`，因此不为表面平衡牺牲主指标，但必须完整披露U/S。

OpenAI CLIP同源Top-4/Top-5图分别为`584/729`边；保留旧438条正式关系句、仅对新增
`146/291`边使用通用类名方向句的Gate A最高为`80.059367/80.077804`，均低于Top-3
正式图。临时通用句不得作为论文证据，当前不生成数百条正式新描述。

R3是R2 checkpoint上的新评估语义，不重新训练reader。代码commit为
`38af1e77dc7fa30b35866e78317b4634a00b9430`，config SHA为
`8528b715c9bc6fcf1f21c4e9da0212cd9efab550efe2c038f24844d7a69766a3`；正式审计复算
仍过全部门后才可保留。

AWA2/SUN图请求资产已提前生成到仓库外。AWA2 Top-3/Top-4/Top-5=`117/159/201`边，SUN
Top-3/Top-4/Top-5=`1633/2180/2729`边；集合manifest SHA为
`4c4491b60bac96ff28555c17cb314baba7fdf5ef2083151499c1696175659dce`。当前资产只包含
准确类别轴、同CLIP类名embedding、关系描述请求与SHA，不包含伪造的正式关系描述。

R3正式签字身份：评估代码`38af1e77dc7fa30b35866e78317b4634a00b9430` + 审查声明tree
`e5b73176e3202513e389bf9b225aa4d0ffe7a538` + config SHA
`8528b715c9bc6fcf1f21c4e9da0212cd9efab550efe2c038f24844d7a69766a3` + source model
SHA `16b5071f21a3217e58a72315029c28b8cfd97b68f812641bd0145d3f5e0702ab` +
environment/GPU fingerprint SHA
`8b3e2d5d93cdd9763843c3c5f72903f466a86f7524c9dc2b02bb1d4699c32c59`。
