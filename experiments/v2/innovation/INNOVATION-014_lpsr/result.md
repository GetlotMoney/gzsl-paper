# V2-INNOVATION-014 结果

状态：`engineering_retry_run002_planned`。

RUN-001在局部文本残差初始化阶段因CPU/GPU设备不一致停止，尚未执行任何参数更新，也未产生实验指标。按项目规范记为`failed_runtime`，不占方法补救次数；保留原输出目录，修复后使用新RUN目录重跑同一条件。

RUN-002仅修复局部文本与类名原型的设备对齐；公式、参数、输入SHA、seed、训练量和评估语义不变。
