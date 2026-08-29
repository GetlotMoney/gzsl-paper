# V2-INNOVATION-065 结果

状态：`rejected_mild_balance_still_overcorrects`。

平方根逆频率把top1/top2类别权重设为`0.732566/2.705670`，但与极低soft gate权重相乘并归一化后，组合权重std仍高达`4.446476`。所有非零条件显著降低H并长期约`77.0%`，best退回父模型与零selector。

完整与平方根两档类别平衡均失败，关闭pair标签平衡轴；原始GWPS保持正式最高。最后补救不再改标签权重，只适度扩大硬margin pair范围。

模型SHA256：`8bca104f7cb7754fc37d3171e9aea2f7e26cd4bf9b87d3cb18e92fa22c4736cd`；最后checkpoint SHA256：`d14c00eaa1ec5f2bdcac88d2716dd7bdccb849aecf71661bf3d1e04f82abfa22`。
