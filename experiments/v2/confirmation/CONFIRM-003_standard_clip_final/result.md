# V2-CONFIRM-003 结果

状态：`completed_owner_authorized_final_evaluation`。

无专家与专家路线分别绑定`V2-TUNE-001/RUN-001`和`RUN-006`的validation选择，从随机初始化在完整`trainval_loc`上重新训练。checkpoint完成后各运行一次official test。

当前方法结构受历史test探索影响，CLIP缓存来源身份不完整，因此结果必须保留`strict_blind_claim_eligible=false`披露。

| Condition | U | S | H | ZS | validation-selected epoch |
|---|---:|---:|---:|---:|---:|
| RUN-001 无专家 | 73.071939 | 76.972061 | 74.971312 | 80.278075 | 24 |
| RUN-002 专家312维属性 | 77.935892 | 79.584587 | **78.751611** | 84.862840 | 22 |
| 专家 - 无专家 | +4.863954 | +2.612525 | **+3.780300** | +4.584765 | - |

两条RUN均从随机初始化开始，每轮完整且唯一遍历7,057张trainval图像；checkpoint写入后各执行一次official test，`test_used_for_selection=false`。专家路线达到`H >= 78%`，无专家路线未达到。

模型SHA：

- RUN-001：`dc211c529763e46781d7f24f5fdd8410acf958fa77516f52fc1039a15c6e8199`
- RUN-002：`2f5c9c324f3f524fd10f5d6bb5623c1f30daa20fde335607ceddac85ba73d498`

本结果不得用于回改当前方法；后续任何新结构都必须回到validation开发，并在另一个未被消费的最终评估上确认。
