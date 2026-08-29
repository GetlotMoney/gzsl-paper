# V3最终视觉资产链审计

## 最终可运行身份

### 正式Linux父CLS

- URI：`/data/lby/projects/cv_project/GZSL_Warehouse/assets/clip_vitl14_336/CUB/openai_d05afc4_codex_roles_v1`
- manifest SHA：`6e54351f1249d1bea1f559d1237ece21450ef0e5d9314df0e863da740df24ec5`
- asset ID：`f61a4af0d7644477`
- raw图像顺序/大小SHA：`97a8fd5596f71b949b89ed1c0b2bb064ad18a358f9613a4e1e7fbe49fa8f2df2`
- OpenAI CLIP Git commit：`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`
- 官方checkpoint SHA：`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
- Linux `clip.py` SHA：`9540f200fbf8145479fa655382a56dab048d238cc698b9cbd8df3b6d86d3f1b6`

### 最终576-patch

- URI：`/data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_projected_patch_final_v1`
- manifest SHA：`d096087c9bd37d90157688e21e79b8ba6a61f0ea9b1fa91f4f544f8bc1dd1ad0`
- asset ID：`cub-clip-patch-e8742e469cd8868e`
- 生成commit：`97f85b013394227edc6fdf6ff9306b5f1f201ab3`
- 公式：`last_resblock_all_tokens -> ln_post -> visual.proj -> L2 normalize`
- shape/dtype：每图`[576,768]`，FP16，逐patch L2归一化。
- split数量：train/test-seen/test-unseen=`7057/1764/2967`。
- CLS最小余弦：train/test-seen/test-unseen=`0.999998808/0.999999762/0.999997020`。
- manifest只记录当前Linux运行环境；不含旧Windows `clip.py`身份。
- manifest包含12个输出SHA，并在继承父CLS文件前后及patch写盘后重新计算。

### 最终36-patch

- URI：`/data/lby/projects/cv_project/GZSL_Warehouse/assets/rgve/CUB_openai_vitl14_336_coarse36_final_v1`
- manifest SHA：`1d60f9a1672c39a04cf7d5fb50dc417736b9fc6d39b81aa4918cb424b8f586c0`
- asset ID：`cub-clip-coarse36-3fbf9cffb754ab6d`
- 父576 manifest SHA：`d096087c9bd37d90157688e21e79b8ba6a61f0ea9b1fa91f4f544f8bc1dd1ad0`
- 公式：`24x24 -> reshape(6,4,6,4) -> mean(block axes) -> 6x6`。
- shape/dtype：每图`[36,768]`，FP16区域均值，消费者加载后L2归一化。
- 文件SHA：
  - train：`4eb9d1c29854bb78996da5fb8f3692db60f836dd5b876cd189c5be599306d4a8`
  - test-seen：`3c590a9988888a189c9429f1f4d81fb9901e6382fe50a8d22e272c41c19f14bc`
  - test-unseen：`e0794c7de3848f0868bf3f60c227cd03143dfd1a8dd9fd42736db170ef4f8b95`

## TG与视觉资产边界

V3-TRY-011保持原TG checkpoint、原文本embedding、原全局CLS和原split不变；视觉分支单独绑定上述正式Linux CLS→576→36链。loader必须同时校验两套资产，并通过逐行标签相等和CLS余弦/最大误差证明第`i`行对应同一图像。`alpha=0`必须精确复现父U/S/H/ZS，否则运行失败。

## 禁止使用的历史资产

以下目录只保留为审计中间产物，禁止后续实验绑定：

```text
CUB_openai_vitl14_336_projected_patch_rebuild_v1
CUB_openai_vitl14_336_projected_patch_rebuild_v2
CUB_openai_vitl14_336_projected_patch_rebuild_v3
```

上述三个`rebuild_v1/v2/v3`目录已由owner明确授权从服务器删除，仅Git审计记录保留；服务器只保留`projected_patch_final_v1`与`coarse36_final_v1`作为后续可用视觉资产。

旧Windows patch manifest的原图指纹算法与生成脚本已经遗失；旧TRY-009/010继续绑定旧资产并保留真实结果，但不得与新资产结果混称同一数据身份。
