# V4-TRY-023 PCLR 单轮双Agent审核 receipt

- 审查对象：`dc2feba264be49a14371444c366c139810ea9445`；tree
  `b1d5a23d6c3af4c6fb68c5e9a8b0955b76ec80f0`；工作树 clean。
- Full config SHA：`3fa1de314be0f31c2f33752239bedf737503ca6e76fd4a45760f5a96b17f3506`。
- Relation manifest SHA：`0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4`；
  `[438,2,768]` FP32关系向量、`[438,2]` int64边，`human_annotations_used=false`，
  `llm_world_knowledge_used=true`。
- 共享本地证据：PCLR/GTD/资产相关`23 passed`，`py_compile`与`git diff --check`通过；
  Reviewer未各自重复整套测试或SHA。
- Reviewer A与B先独立完整审查，再直接交换完整发现并互相质询一次。静态共同结论：
  `P0=0/P1=0`，待真实micro。
- 同一身份GPU0 micro覆盖两步combined backward、Parent影子参数/损失/RNG同轨迹、
  official GZSL/ZS、GTD-Off、teacher package及checkpoint恢复后下一步完全一致；
  `checkpoint_next_step_equal=true`，update-2 state SHA
  `081a2db5711a1b521fd2ca00e546e958961853a6606c40df16c78f30e82a176a`。
- 物理GPU1完成relation+beta真实前反向，loss=`3.783186197280884`，梯度全部有限。
- 两名原Reviewer并行复核micro后共同结论：`P0=0/P1=0`，
  **代码单轮双Agent对抗审核通过**，允许启动唯一direct-official Full。
- P2：关系文本与Parent为同一官方CLIP checkpoint但`clip.py` SHA不同，manifest已明确；
  Full后须逐update比较PCLR-Off与RUN-030的152点；finalize尾部恢复窗口与输出目录由运行
  纪律管理。这些不阻断启动，但不得在结果解释时省略。

## Post-run审计

- Full完成：history=`152`，stop reason=`completed_fixed_150`，code/config/asset及所有结果
  文件SHA均通过；best `U/S/H/ZS=75.926912/82.386708/79.025018/87.248874`，update
  `13818`，相对Parent `-0.044997 H`，同checkpoint Off `+0.123105 H`，net=`4`。
- Parent/GTD最终34个张量、optimizer与scheduler均与RUN-030 bitwise一致；152点Off
  U/S/H全部一致。唯一偏差为update `18471`的Off-ZS：`86.734289` vs `86.774284`，
  根因为PCLR Off与旧`_predict`的prototype normalize语义不完全相同。
- 两名原Reviewer并行裁决：RUN记为`engineering_failed_reporting_deviation`，不可作为
  canonical论文结果；方法三个成功门均失败，记为`drop_no_rerun`。Parent增益为负，不
  触发预注册cap救援；不跑controls、不重训。
