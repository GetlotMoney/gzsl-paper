# V2-INNOVATION-098 结果

状态：`rejected_nonlinear_seen_ce`。

错误审计：S-EDPS可处理top2错误中，seen仅61/155、unseen仅110/288获得正确负delta方向；全局放大delta会降低H，瓶颈是线性交互能力而不是幅度。

唯一改动：冻结S-EDPS的12维线性selector，在其raw score上增加`12→8→1`零初始化受限残差，最大raw残差`0.25`。只训练残差MLP；关闭时逐位复现S-EDPS。seed5须超过`78.572828%`才追加seed7。

RUN-001完整训练后best严格保持S-EDPS父模型`U/S/H/ZS=76.982599/80.230141/78.572828/84.121776%`，selected iteration=`-1`。最后checkpoint的MLP输出层范数=`0.886818`、bias=`-0.089341`、第一层范数=`1.982651`，证明残差真实训练但所有非零状态均更差。

结论：方向错误不是线性容量不足，而是seen pair CE对unseen方向缺乏可迁移监督；不继续缩放hidden或残差上限。模型SHA256：`de5d78599a1a91f8ca2b6ca2272462ba0ac01d8eb5fe8f2dc59b56e461136920`；最后checkpoint SHA256：`fd82ccd52656bbf4cd3659abac74791aafce04f197b4971c560a06f90749b44c`。
