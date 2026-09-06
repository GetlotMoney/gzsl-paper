# V4-TRY-023-R3 PCLR推理调参单轮双Agent审核 receipt

- 冻结评估代码：`38af1e77dc7fa30b35866e78317b4634a00b9430`；tree：
  `ab2e027d417adbd6128dce398f8cbe2ac1499213`。
- Config SHA：`8528b715c9bc6fcf1f21c4e9da0212cd9efab550efe2c038f24844d7a69766a3`。
- Source R2 code/config/model/metrics/Off-history SHA分别绑定为`b0a756dd...`、`0861877...`、
  `16b5071...`、`3d64bd3...`、`d5d7049...`；source只读。
- 共享本地证据：相关`27 passed`，`py_compile`与`git diff --check`通过。

Reviewer A/B先独立审查source身份、训练/推理温度分离、Top17/ridge0.3/cap0.5、
scale6.95/gamma0.575、Raw/Calibrated/Full三路、ZS晚切、macro per-class指标、AND成功门、
原子输出与nested official-test披露，再直接交换完整清单并质询。静态共同`P0=0/P1=0`。

物理GPU0/GPU1各自完成全部测试样本的完整micro，metrics逐字节一致，SHA均为
`39bea2dbf664dc421cd53b2a4f8d219b85f05b9279e7991b596c61d22aa4042a`。共同结果：

- `U/S/H/ZS=77.806163/82.716906/80.186419/87.612945`；
- `ΔH`相对Parent/Raw Off/Calibrated Off=`1.116404/1.284507/1.692670`；
- gap=`4.910743`，seen/unseen net=`68`，active edge rate=`0.0370702`；
- effective beta=`0.725859`，effective beta max=`1.7375`；
- 六项成功门全部满足，`full_gate_passed=true`；正式输出不存在。

两名Reviewer直接互认micro，最终`P0=0/P1=0`，共同结论：

**代码单轮双Agent对抗审核通过**

P2：不保存逐类accuracy向量；snapshot写入非原子；TopK exact tie；source state必须与config
共同发布；effective beta较大必须持续披露。R3明确
`nested_official_test_selection=true/test_used_for_hyperparameter_selection=true/strict_blind_claim=false`，
不得包装为blind结果。P2不阻断本次已复算结果。
