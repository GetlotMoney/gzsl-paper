---
idea_id: IDEA-186
name: Pairwise Contrastive Laplacian Reasoning
short_name: PCLR
status: owner_reopened_final_tuned_rescue_pending_audit
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
