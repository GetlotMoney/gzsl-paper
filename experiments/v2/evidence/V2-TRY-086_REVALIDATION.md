# V2-TRY-086 训练规范复核

复核日期：2026-08-23

## 结论

`V2-TRY-086 / CCGR + SDM / seed 17`符合项目固定的
`test_selected_inductive_gzsl`协议，可以保留为当前无专家属性路线的可复现结果。

- 训练梯度只使用CUB `trainval_loc`的7,057张图像和150个seen类。
- 三折中的pseudo-unseen来自上述150个seen类内部划分，不是真实50个unseen类。
- 真实unseen图像不参与梯度。
- official test-seen/test-unseen参与逐epoch选模，记录为
  `test_used_for_selection: true`，因此本结果不是blind-test。
- 代码链未读取CUB 312维人工专家属性；`att_splits.mat`只用于核对官方划分、标签和类别顺序。

## 数据边界复核

服务器缓存与xlsa17逐项核对结果：

| split | 样本数 | 类别数 | 核对结果 |
|---|---:|---:|---|
| trainval | 7,057 | 150 | 缓存标签与`trainval_loc`逐项一致 |
| test-seen | 1,764 | 150 | 缓存标签与`test_seen_loc`逐项一致 |
| test-unseen | 2,967 | 50 | 缓存标签与`test_unseen_loc`逐项一致 |

三个原始索引集合两两不相交，训练类别与真实unseen类别集合不相交。

## 全链复跑

复跑输出：

`/data/lby/projects/cv_project/GZSL_Warehouse/revalidation/v2/V2-TRY-086-FULLCHAIN-20260823`

复跑依次覆盖基线、三折模型、NTR、CCGR和SDM。各阶段使用其原始准确commit与配置；
下游仍绑定原始父checkpoint，但每个新训练父模型均与原父模型字节级一致，因此两条依赖链数值等价。

| 阶段 | code commit | 复核模型 | SHA256 |
|---|---|---|---|
| TG-VPR基线 | `3dc078c0d52bf358bf24a26e48346c97de9e99ca` | `model_best.pth` | `59397353a7db1b82df815a0faea1050c76b6f992ff3426a2bbb58d984f39595f` |
| ELPT fold 0 | `74594447c4eff4cd63058113d6d297ee08a0e4ee` | `fold_0.pth` | `a0a255353e36f57167c5c4fc685e2ccd253f12fb4b3003f9672c7a5d7dd0f09b` |
| ELPT fold 1 | `74594447c4eff4cd63058113d6d297ee08a0e4ee` | `fold_1.pth` | `0fe6490e5dde54ab28bf873f7062dad8e1b5ed7d52da03a00d1dc11325d8c372` |
| ELPT fold 2 | `74594447c4eff4cd63058113d6d297ee08a0e4ee` | `fold_2.pth` | `de42fa0be7aec88d83cf88f9ed0ec8cd5c8211f8971ed5307bcc0aaaf2c86d2b` |
| ELPT gate | `74594447c4eff4cd63058113d6d297ee08a0e4ee` | `gate_model.pth` | `d4176425d465e9d7e04642f7718f15393d918dde3121a2296d8a07cb11b59fa6` |
| NTR | `42cd4457a65f89023ff342ba13679471d5db0942` | `gate_model.pth` | `8f64f8bec9e801af29fc46b46a61a859243745a3ca99c741037577b2457ddebb` |
| CCGR | `12e8c99bb377e9b31ba7621575c2d9de498027fd` | `ccgr_model.pth` | `5a7a53dd0a8674b6b088afe023e63de0a3a50a5f4a21cf72ba370973a20af9e4` |
| SDM | `75fb252606fa4a0a5f0709e78053fe156668596c` | `sdm_model.pth` | `e1aa1342b05cfb8d319cd87e5708548a97cf450eb9fb179e8a5acb80f77748d6` |

上述8个新模型文件均与原模型文件字节级一致。

## 复跑结果

| U | S | H | ZS | best epoch |
|---:|---:|---:|---:|---:|
| 74.652368 | 80.818135 | **77.612988** | 82.173079 | 2 |

SDM相对CCGR父条件的H增益只有`+0.040306`个百分点，因此它是可复现的辅助优化，
不是足以单独支撑核心创新的提升；无专家属性路线仍未达到`H >= 78%`。

## 验证环境

- 服务器：`lab4090`
- 物理GPU：1（RTX 4090）
- 复跑前后服务器仓库工作树干净
- 服务器已恢复到复跑前HEAD：`6435b39b7afd8ff2548dcc978cbe51364b2b38e6`
- 相关测试：36项通过；服务器未安装pytest，pytest风格的21项无fixture测试由直接调用执行
