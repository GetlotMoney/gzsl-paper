# V6-TRY-006 C-PCLR代码审查记录

> 本页记录的`89b2908...`独立审查已被200-epoch matched-control语义修订取代，仅保留历史。
> 最终直接文件交换审查将在新冻结commit产生后追加，旧签字不得用于RUN。

## 冻结身份

- 最终代码commit：`89b2908a388c4c6586bbf19216fd50fc777ffdb3`
- 准确父commit：`52b511d77b4ad048f35b40dc3cbd9afd092167e9`
- RUN config SHA256：`ea8bb8af16b269daeb45a9482d92d3222ef796733acc944e45ed32ac28461faa`
- 资产manifest SHA256：`3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6`
- 关系资产manifest SHA256：`0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4`
- 本地最小验证：`tests/test_pclr.py + tests/frameworks/v6/test_compiled_pclr.py = 29 passed`
- GPU fingerprint：pending micro-batch

## 独立审查与集中修复

两名临时Reviewer始终只读审查同一冻结commit；主Agent只在两份完整清单均返回后集中修复。

1. `175ae9cba4578c1b9dd943e6112d64a72480a166`
   - A：`P0=0/P1=2/P2=4/revise`；update0可keep、缺独立best-ZS。
   - B：`P0=0/P1=1/revise`；同意update0不能证明训练内化。
   - 主Agent补充P1：从已训练R2 checkpoint继续21171步违反同预算与一段式合同。
2. `c55cf2243c23675242ede54385ffd5d18e0e7941`
   - 一次性修复为同seed7初始化、TG/GTD与C-PCLR同一21171步训练；补update0门、best-ZS、
     原子checkpoint和非零head梯度收据。
   - A/B发现source梯度收据误判关闭模块参数；B另发现train-mode dropout污染prototype同步。
3. `23758d455ce8d9ee38f01e66865eca1754b96862`
   - 修复为只检查启用参数组；prototype同步临时eval、RNG中性并恢复训练态。
   - A/B发现`semantic_group_logits`兼容参数仍混在active组且恒为`grad=None`。
4. `89b2908a388c4c6586bbf19216fd50fc777ffdb3`
   - 每个非空active组要求至少一个实际有限梯度，允许组内未参与forward的兼容参数无梯度；
     新增真实`PaperV2ThreeModuleModel(full/off/off)`测试。
   - Reviewer A最终：`pass，P0=0/P1=0/P2=2`。
   - Reviewer B最终：`pass，P0=0/P1=0/P2=2`。

## 直接交叉状态

两名Reviewer均明确报告其临时子Agent工具集中没有`collaboration.send_message`或等价的子Agent
互传工具，无法直接向对方发送完整清单或逐项回应。主Agent没有用转述冒充直接交叉。

因此当前准确状态为：`两份独立pass，但直接交叉未完成`。在owner明确批准协议例外前，不得写
“双Agent交叉审查通过”，不得启动GPU micro-batch或正式RUN。
