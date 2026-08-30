# V4-TRY-013 result

状态：`gate_passed / keep_for_formal_gzsl`。

预注册输出：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v4/prequeue/IDEA-182-pecv-gate-seed7`。

## 结果

| 条件 | 50类class-disjoint Macro Top-1 |
| --- | ---: |
| Frozen Parent | 67.9755% |
| PECV-Full | 72.5250% |
| PECV-Shuffled semantics | 32.0389% |

- 相对Parent：`+4.5495pp`，预注册门槛`+1.0pp`。
- 纠正/损坏/净纠正：`121 / 12 / +109`。
- 整体语义打乱损失：`40.4862pp`，预注册门槛`0.5pp`。
- Parent Top-5覆盖率：macro `95.3451%`，micro `95.2866%`。
- `module_off_exact_parent=true`。
- `eval_images_used_for_gradient=false`。
- `checkpoint_roundtrip_verified=true`。

## 身份

- RUN commit：`6043e1d7cc5c1af219a7dc0a952398faff27c91f`
- config SHA：`10ddca9dbc22cffb6f13dc3254e011826d84387943268e613a35c1a8b54e4a50`
- metrics SHA：`02f8264584f2704bac7898caa5340eeee3e6a276d2e5a5f4546a4ed42d7ab7f7`
- evidence SHA：`423df9834889387b0d365fbfb00f92f53f5fcd0bffbe567ea6013583ae844473`
- checkpoint SHA：`d1dfb081ebea77c6fb88515905e3494bddc6a8eeef874595721cab28bb04666a`

## 结论边界

PECV已证明“100类学到的候选差异裁判可以迁移到隔离50类并改善Parent候选内排序”。它尚未进行正式200类GZSL联合竞争，因此本结果不能当作U/S/H/ZS，也不能据此晋级`framework/v5`。
