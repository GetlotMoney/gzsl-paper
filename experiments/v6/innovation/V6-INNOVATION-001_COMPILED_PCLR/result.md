# V6-INNOVATION-001 / V6-TRY-006 result

状态：`completed_drop_below_parent`。两名Reviewer对commit
`b707b0c4671051244cebf4f8404299fc016b281e`完成独立清单和直接文件交换，双方最终
`P0=0/P1=0/pass`；GPU batch50 micro-batch通过。正式RUN在commit
`8de7cebda0235ab12e1b4b8f669134c8f4e2c075`完成200名义epoch、28,228 updates和202行
official评估历史。

预注册判断：Full必须同时高于正式FRAMEWORK-V5 `H=81.06877662507551`和同seed、同28,228步
训练的matched online-V5 control最佳H；同一Full
checkpoint下S-off、V-off、I-off各自必须使H至少降低`1.0pp`，且`|U-S|<8`。任一失败即drop，
不启动Top-K、ridge、scale、gamma、seed或checkpoint补救搜索。

固定披露：official test用于整模型checkpoint选择和本候选配置确认；
`unseen_images_used_for_gradient=false`，`strict_blind_claim=false`。

## 正式结果

best-Full-H checkpoint=`update13818`：

| 条件 | U | S | H | ZS | Full−off H |
|---|---:|---:|---:|---:|---:|
| C-PCLR Full | 77.606910 | 83.639657 | 80.510432 | 88.473403 | — |
| S-off | 76.141131 | 82.428479 | 79.160157 | 87.064338 | 1.350275 |
| V-off | 82.206428 | 76.821315 | 79.422694 | 88.181764 | 1.087737 |
| I-off | 82.451552 | 76.188660 | 79.196481 | 88.189560 | 1.313951 |

- matched online-V5 best：`U/S/H/ZS=80.112976/81.535739/80.818096/88.646406`
  `@ update13818`。
- 正式FRAMEWORK-V5：`H=81.068777`。
- C-PCLR相对matched control：`-0.307664 H`；相对正式V5：`-0.558345 H`。
- C-PCLR独立best-ZS=`88.473403 @ update13818`；matched control独立best-ZS=
  `88.956916 @ update4935`，未与best-H拼接。
- 三个module-off硬门全部通过；Full父条件门失败。

程序decision=`drop_gate_b_contract_failed`。结果表明编译关系、视觉Reader和角色语义均形成
真实deployment dependency，但简单冻结出口没有保留在线V5的全部准确率。按预注册停止该Idea，
不运行Gate C或任何补救参数搜索。

## Owner覆盖决定

owner在查看完整结果后明确决定“这次就算通过”，因此本候选保留为
`owner_override_keep_efficiency_candidate`。该决定不改写程序原始drop和真实差值：它表示owner
接受`-0.307664 H`相对matched online-V5、`-0.558345 H`相对正式V5的准确率损失，并准备从
“精度更高”改为“精度—部署成本折中”评估。真实速度、显存和吞吐尚未测量；补齐之前不得写
“更高效”“Pareto优势成立”或“已晋级正式framework”。

## 仓库外证据

- URI：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v6/compiled_pclr/V6-TRY-006`
- config snapshot SHA：`13a16b85aee055b42024d7645c89c5e4ccf26800d84c9b438760c0ab1cd3bec0`
- metrics SHA：`fbbd8ef520d8d6bca62cc1d860a0432a244ab99af30761a3ffd8c824f7c90879`
- evaluation history SHA：`746e02c8dfc40d21fba1ad1c9051502d545e4cd75eb4cdc2d3dd8cbd2fcbcd96`
- model best SHA：`a551de9d182222141ab4be9db1ae2020417be3a7a7d1d4b369510d635f2207c9`
- training log SHA：`d0ba36c136beb9249ca699f25a237d0060c801522ff728f625eaf4a6820f75b7`
