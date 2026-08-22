# 实现说明

JBEC加载每个seed冻结的VEBC父beta/gamma，只训练范围`±2`和`±0.05`的零初始化残差。fold内正反ridge仅使用pseudo-seen中心，真实unseen图像不进入梯度。
