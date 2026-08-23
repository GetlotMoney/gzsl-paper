# PATCH_CACHE_PROVENANCE_AUDIT_001

审计范围仅限当前`gzsl-paper`仓库与服务器当前项目数据入口，未读取旧GTPJ。

## 已确认事实

- train patch：`[7057,576,768]`，SHA256=`244dbf96109362306555c55cf718590a6ffd2f3c5c403127463aff86311aacc2`。
- test-seen patch：`[1764,576,768]`，SHA256=`d1a21b3a4797ffa28115ffb22cc6d1e401453d605b99360c731d46e523a794ca`。
- test-unseen patch：`[2967,576,768]`，SHA256=`3ef523a1fe582fa2e1be88f3a975bda64e31a565e69fe001efec2f21fa3b1d77`。
- 三个patch文件与对应CLS/label缓存在`2026-06-28`连续写入，属于同一批缓存产物。
- 当前仓库只有缓存加载、形状检查和实验消费代码，没有原图→patch缓存生成脚本。
- 当前项目数据挂载只有`data/cache`与`data/xlsa17/data/CUB`，没有CUB原图，因此无法从候选checkpoint重提样本做逐值比对。

## 不能证明的内容

- `576=24×24`且输出768与336输入的ViT-L/14类CLIP相符，但不能据此证明具体实现、权重来源或预处理。
- 无法确认是OpenAI CLIP、OpenCLIP或其他兼容checkpoint。
- 无法确认resize/crop、图像归一化、patch token投影和最终归一化细节。

## 决策

所有依赖该patch cache的结果继续标记`feature_provenance_complete=false`，包括当前最高GWPS。不得把推断的backbone名称写成事实或用作严格同骨干外部比较。patch-free AGCT/T-GWPS结果继续作为可披露对照。
