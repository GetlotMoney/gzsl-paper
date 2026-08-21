# TG-VPR-H1：正式创新模块1

## 身份

- 正式名称：Tri-Group Value Prototype Reparameterization，简称`TG-VPR-H1`
- 仓库身份：`INNOVATION-MODULE-1`
- 状态：机制与多seed结果已验证，作为独立模块迁入
- 集成状态：尚未接入`FRAMEWORK-V1`主训练入口，因此不会静默改变V1
- V1基线关系：当前确认基线代码为`f8dd7c72465686cfe4aea8a0f37f658e1176386a`，H=`74.2468`
- Innovation编号：尚未分配；`V1-INNOVATION-001`继续保留给满足本仓库Idea/PAPER证据门的正式实验

本次只迁入方法、代码、配置和最小结果摘要；不迁移旧GTPJ的论文笔记、idea编号、实验目录或Git历史。

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

这些结果使用official test做结构选择，属于`test_selected_inductive_gzsl`下的test-exposed证据，不是blind-test或独立confirmation。

## 代码与配置

- 模块：`model/tg_vpr_h1/module.py`
- 独立训练入口：`python -m model.tg_vpr_h1.train`
- 冻结配置：`config/tg_vpr_h1.yaml`
- 来源：`docs/TG_VPR_H1_SOURCE.yaml`
- 测试：`tests/test_tg_vpr_h1.py`
- HTML框架图：`docs/TG_VPR_H1_framework_diagram.html`
- 可编辑源图：`docs/TG_VPR_H1_framework_diagram.drawio`
