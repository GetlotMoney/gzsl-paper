# PAPER-006：Universal Prototype Transport

```yaml
paper_id: PAPER-006
title: Universal Prototype Transport for Zero-Shot Action Recognition and Localization
authors: Pascal Mettes
year: 2023
venue: International Journal of Computer Vision
arxiv: 2203.03971
publisher_url: https://link.springer.com/article/10.1007/s11263-023-01846-2
source_checked_at: 2026-08-23
pdf_uri: G:\CV-Workspace\Literature\gzsl-paper\PAPER-006_universal_prototype_transport.pdf
pdf_sha256: 24c051a6f84c2e28699d169a23ac2e20acc388984ad0fe76081657dcc745c8c2
```

## 与本项目关系

论文通过超球面最优传输得到目标原型，并沿原始与目标原型的测地线重定位unseen原型。它是TST最接近的几何先例。关键差异是UPT使用完整test视频分布、属于transductive action ZSL；TST不读取true-unseen图像，只由目标类文本Value方向和seen内部pseudo-unseen episode训练步长。
