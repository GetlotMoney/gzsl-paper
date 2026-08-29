# FRAMEWORK-V2 三数据集最终实验协议

当前论文实验固定使用`CUB / AWA2 / SUN Attribute`三套Xian Proposed Split。SUN明确指717类、14,340张图像的SUN Attribute数据集，不是SUN397。

## 数据身份

| Dataset | classes | seen/unseen | trainval | test-seen | test-unseen |
|---|---:|---:|---:|---:|---:|
| CUB | 200 | 150/50 | 7,057 | 1,764 | 2,967 |
| AWA2 | 50 | 40/10 | 23,527 | 5,882 | 7,913 |
| SUN | 717 | 645/72 | 10,320 | 2,580 | 1,440 |

原始图像、八角色原文、CLIP缓存和运行产物全部位于服务器Warehouse，不提交Git。三个数据集统一使用OpenAI CLIP `ViT-L/14@336px`官方权重，checkpoint SHA256固定为`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`。

正式缓存统一称为`canonical_visual_plus_role_text_v2`：每个数据集的资产包同时包含全局视觉特征、train/test标签、类别顺序和第二版八角色文本特征。旧称`text_v2`只描述文本组件的版本，不代表资产包缺少视觉特征。带patch、多层或增强视图的超集包统一称为`canonical_visual_plus_role_text_v2_with_dynamic_visual_extensions_v1`；模型可以只读取所需字段，未读取的扩展字段不改变计算语义。旧asset ID、目录和历史实验引用保持不变，用于追溯。

## 条件与模块

```text
B0 Pure CLIP = image × "a photo of a {class name}"
B1 Mean8     = image × mean(eight role descriptions)
M1           = B1 + TG-VPR
M2           = M1 + TST-NTR
M3           = M2 + CCGR
```

八角色接口统一为`[class_count,8,768]`；前六句由数据集角色定义，第七句为overall，第八句为unique。CUB、AWA2和SUN使用各自视觉角色，但共享相同张量接口、模块公式和训练代码。

## Chen-style边界

每个数据集使用全部`trainval_loc`图像训练。每步独立执行`randperm(ntrain)[:50]`，总更新数为`ntrain×200//50`，每`niters//200`步评估official test并根据同一checkpoint的H保存整模型best。固定披露：

```yaml
test_used_for_selection: true
test_used_for_hyperparameter_selection: true
unseen_images_used_for_gradient: false
strict_blind_claim: false
nested_official_test_selection: false
```

Stagewise主策略固定为50/100/50名义epoch：先训练TG-VPR，再冻结TG训练TST-NTR+CCGR，最后全部解冻低学习率联合微调。End-to-End使用相同总更新数作为训练策略消融。阶段边界不读取test决定，阶段间只传最后权重，整次RUN只有一个全局best-H。

## 入口与制品

- 数据与split：`tools/gzsl_data.py`
- 八角色请求：`tools/create_role_text_request.py`
- CLIP缓存：`tools/prepare_paper_clip_assets.py`
- 三模块模型：`model/frameworks/v4/model.py`
- 历史三模块训练器：原路径`model/train_paper_v2.py`，以各Experiment绑定的准确code commit为准，当前`main`不保留候选入口
- V4晋级来源复现：`python -m model.frameworks.v4.train`
- 历史module-off：原路径`tools/evaluate_paper_module_off.py`，仅在对应实验commit中存在

每个正式RUN至少生成`training.log / metrics.json / model_best.pth / checkpoint_last.pth / data_fingerprints.json / config.snapshot.yaml / evaluation_history.json`。
