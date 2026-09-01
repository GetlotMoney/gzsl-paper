# V6-TRY-006 GPU micro-batch收据

- 执行日期：2026-09-02
- 运行worktree commit：`3dd338cd7b57b2a3afd31c81c68e40cc3d93f5cd`
- 审查代码commit：`b707b0c4671051244cebf4f8404299fc016b281e`
- review目录tree：`12a0961f4d8f59d560ab4526976a51ef2771ed1b`
- config SHA256：`73a812268b18e9f46a2cedf59acdabb8ef0cdb13388ec83b5f23b73475e4239b`
- 资产manifest SHA256：`3a6b261a63e2aa241d7a9cd2b3c9b0051a0ba01133ef61dc35e0d043fc119fa6`
- 关系资产manifest SHA256：`0d94188e895fb1c2034233f6562682cf31ba04ea1f3f504fc30d7f0643e143c4`
- micro物理GPU：index0，`NVIDIA GeForce RTX 4090`，UUID
  `GPU-1dca1cb0-d2a2-c075-af6e-a3e9a1eeb968`，driver `525.147.05`，24564 MiB。
- batch：50；同一batch执行2个内存更新以越过Reader-out零初始化并验证Reader-in真实非零梯度。
- persistent writes：false。

## Loss

- joint total：`10.279571533203125`
- Parent CE/topology/gate：`3.0656466484 / 0.0214632750 / 0.0777126402`
- matched control relation/beta：`0.5117341280 / 3.3423027992`
- C-PCLR classification/relation：`2.7639250755 / 0.5118114948`

## 关键梯度

- C-PCLR `raw_alpha=0.08046262`，`raw_role_weights=3.30755711`
- C-PCLR Reader-in weight/bias=`0.00242091 / 0.00283285`
- C-PCLR Reader-out weight/bias=`0.09870086 / 0.51067400`
- matched control `raw_beta=0.01717208`
- matched control Reader-in weight/bias=`0.00165383 / 0.00191861`
- matched control Reader-out weight/bias=`0.06650916 / 0.34279826`
- TG/GTD active parent/gate梯度全部有限；`semantic_group_logits`兼容无梯度不误阻断。

## 导出与裁决

- `Q` shape=`[200,1536]`
- `bias` shape=`[200]`
- 所有loss、梯度、alpha、角色权重和导出张量均有限。
- micro-batch：`pass`。

正式RUN预注册使用空闲物理GPU1：UUID
`GPU-b1df9ad6-832e-1cb8-f096-d49380875928`，通过`CUDA_VISIBLE_DEVICES=1`映射为配置中的
逻辑`cuda:0`。正式环境fingerprint SHA256：
`027e187f11992788ab8b5e780ab4fd709b9935255d47d652bae7cb4335494b69`。
