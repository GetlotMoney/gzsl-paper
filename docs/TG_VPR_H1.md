# FRAMEWORK-V2：TG-VPR-H1

## 身份

- 正式名称：Tri-Group Value Prototype Reparameterization，简称`TG-VPR-H1`
- 正式框架：`FRAMEWORK-V2`
- 来源身份：`INNOVATION-MODULE-1`
- 冻结分支与Tag：`framework/v2`、`v2`
- 状态：owner已将独立模块提升为V2；当前仓库首个正式单seed基线已完成
- 集成状态：V2拥有独立训练入口，不接入或修改`FRAMEWORK-V1`
- V1基线关系：当前确认基线代码为`f8dd7c72465686cfe4aea8a0f37f658e1176386a`，H=`74.2468`
- V1 Innovation编号：不占用；`V1-INNOVATION-001`继续保留给V1下的正式创新实验

本次只迁入方法、代码、配置和最小结果摘要；不迁移旧GTPJ的论文笔记、idea编号、实验目录或Git历史。

## 当前仓库正式基线

`V2-CONFIRM-001 / RUN-001`使用seed 7得到：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`，best epoch=`50`。准确代码为`3dc078c0d52bf358bf24a26e48346c97de9e99ca`。

该H逐值复现历史seed 7，但当前仓库只完成了一个正式seed，不能据此声称新的多seed稳定性结论。

## 方法

八句话按语义组成三个平级证据组：前六句局部描述取均值，第八句作为独特特征，第七句作为整体外观。三个组固定各占`1/3`。

对第`k`组语义`g_k`，单一768维Value路径产生上下文`c_k`，内部残差为：

\[
\tilde g_k=\operatorname{LN}\left(2\left[0.35c_k+0.65g_k\right]\right).
\]

最终seen类原型为：

\[
p=\operatorname{Norm}\left(0.35b+\frac{0.65}{3}\sum_{k=1}^{3}\tilde g_k\right).
\]

训练目标为：

\[
\mathcal L=\mathcal L_{CE}+0.1\mathcal L_{topology}.
\]

其中topology loss约束重参数化前后200类原型的两两关系，防止Value改写越强时seen类几何边界过度漂移。unseen类始终保留原始Mean8原型。

## 已验证结果

固定等权条件在CUB seeds `5/6/7/8`上的H分别为：

```text
73.709453 / 73.881597 / 74.023182 / 73.798142
```

H mean=`73.853094`，min=`73.709453`，max=`74.023182`，range=`0.313729`。固定`1/3`比旧可学习权重四seed平均高`0.014826 H`；该微小差异不作为性能claim，只说明删除可学习组权重没有代价。

这些迁入结果使用official test做结构选择，属于`test_selected_inductive_gzsl`下的test-exposed历史证据，不是blind-test，也不替代本仓库V2正式基线。

## 代码与配置

- 模块：`model/frameworks/v2/model.py`
- 独立训练入口：`python -m model.frameworks.v2.train`
- 冻结配置：`config/tg_vpr_h1.yaml`
- 来源：`docs/TG_VPR_H1_SOURCE.yaml`
- 测试：`tests/test_tg_vpr_h1.py`
- V2身份：`experiments/v2/FRAMEWORK.yaml`
- V2模块说明：`experiments/v2/MODULES.md`
- HTML框架图：`experiments/v2/framework_diagram.html`
- 可编辑源图：`docs/TG_VPR_H1_framework_diagram.drawio`
- 直接迁入的旧实验轻量证据：`experiments/v2/evidence/legacy_h1/`
