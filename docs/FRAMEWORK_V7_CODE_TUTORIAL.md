# FRAMEWORK-V7 从零开始：PyTorch、图拉普拉斯与代码详解

这份教程面向第一次接触 PyTorch、GZSL 和图拉普拉斯的读者。目标不是让你背公式，而是让你能回答下面五个问题：

1. 一张图片为什么会变成一个长度为 768 的向量？
2. PyTorch 如何训练 Reader、关系强度和角色权重？
3. 438 条“类别 A 相对类别 B”的文本关系，如何变成 200 个类别原型？
4. 为什么训练时需要关系图，部署时却只需要 `Reader + Q + b`？
5. V7 的 U、S、H、ZS 和 S/V/I 关闭实验究竟在测什么？

建议第一次按顺序阅读；以后查代码可以直接跳到“代码地图”。

---

## 1. 先用一句话理解 V7

V7 做的事情是：

```text
训练阶段：
图片 + 类别文本 + 类别之间的比较文本关系
→ 学会基础类别原型、角色语义、视觉关系读取和全局关系原型

部署阶段：
图片特征 x
→ Reader(x)
→ 拼接成 h(x)
→ logits = h(x) Qᵀ + b
→ 在 200 个类别中选分数最高者
```

训练时出现的 438 条边、incidence 矩阵和拉普拉斯求解，已经被编译进固定类别矩阵 `Q`。部署时不再逐图片执行图算法。

---

## 2. PyTorch 是什么

### 2.1 为什么需要 PyTorch

神经网络训练的本质是：

1. 用一组带参数的公式计算预测；
2. 计算预测与正确答案之间的误差；
3. 求“每个参数应该往哪个方向调整”；
4. 小幅更新参数；
5. 重复很多次。

PyTorch 帮我们完成两件麻烦事：

- 快速执行大量矩阵运算；
- 自动计算梯度，也就是误差对每个参数的导数。

V7 代码中的核心导入是：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

- `torch`：张量和矩阵运算。
- `torch.nn`：神经网络模块，例如线性层。
- `torch.nn.functional`：无状态函数，例如 `normalize`、`gelu`、`linear` 和交叉熵。

### 2.2 Tensor：带形状的数字容器

Tensor 可以理解为“允许放到 GPU 上高速运算的多维数组”。

```python
x = torch.randn(50, 768)
```

这表示：

- 当前 batch 有 50 张图片；
- 每张图片用 768 个数字表示；
- `x.shape == [50, 768]`。

V7 中常见张量：

| 名称 | Shape | 含义 |
|---|---:|---|
| `x` | `[B,768]` | B 张图片的 CLIP CLS 特征 |
| `P` | `[200,768]` | 200 个基础类别原型 |
| `R` | `[200,8,768]` | 每类 8 个角色语义原型 |
| `D` | `[438,768]` | 438 条边的有向文本差异 |
| `B` | `[438,200]` | 边和类别节点的 incidence 矩阵 |
| `M` | `[200,438]` | 把边信息转换到类别节点的映射 |
| `G` | `[200,768]` | 编译后的类别关系原型 |
| `h` | `[B,1536]` | 原图特征与 Reader 特征的拼接 |
| `Q` | `[200,1536]` | 最终 200 类分类矩阵 |
| `logits` | `[B,200]` | 每张图片对 200 类的分数 |

### 2.3 矩阵乘法 `@`

代码：

```python
logits = h @ Q.T
```

Shape 变化：

```text
h      : [B,1536]
Q      : [200,1536]
Q.T    : [1536,200]
logits : [B,200]
```

对一张图片来说，`h @ Q.T` 等价于把图片向量分别与 200 个类别向量做点积。点积越大，表示图片与该类别越匹配。

### 2.4 点积是什么

两个三维向量：

```text
x = [1, 2, 3]
y = [4, 5, 6]
```

点积：

```text
x·y = 1×4 + 2×5 + 3×6 = 32
```

CLIP 分类也是同一个原理：图片向量与类别文本向量越同方向，点积越大。

### 2.5 `F.normalize`：只比较方向

```python
image = F.normalize(image_features, dim=-1)
```

归一化会把向量长度变成 1：

```text
normalize(x) = x / ||x||₂
```

这样点积主要表示方向相似度，而不是被向量长度左右。

### 2.6 `nn.Module`：一个可组合的模型部件

```python
class V7DeploymentModel(nn.Module):
    def forward(self, image_features):
        ...
```

继承 `nn.Module` 后，PyTorch 能自动管理：

- 模型中的参数和 buffer；
- CPU/GPU 转移；
- 保存和加载 `state_dict`；
- 调用 `model(x)` 时自动执行 `forward(x)`。

### 2.7 Parameter 和 Buffer 的区别

训练参数使用 `nn.Parameter`：

```python
self.raw_alpha = nn.Parameter(torch.tensor(...))
```

它会：

- 出现在 `model.parameters()` 中；
- 接收梯度；
- 被 optimizer 更新。

固定张量使用 buffer：

```python
self.register_buffer("q", q, persistent=True)
```

它会：

- 跟随模型一起移动到 GPU；
- 保存在 `state_dict` 中；
- 默认不被训练更新。

V7 部署模型里的 `Q`、`b` 和 Reader 权重已经冻结，所以全部注册为 buffer。

### 2.8 `detach()`、`cpu()`、`float()`、`clone()`

V7 初始化中有：

```python
torch.as_tensor(q).detach().cpu().float().clone()
```

逐个解释：

- `as_tensor`：把输入变成 PyTorch Tensor；
- `detach`：切断旧的梯度关系；
- `cpu`：先复制到 CPU，形成稳定的加载边界；
- `float`：统一成 float32；
- `clone`：拥有独立内存，避免外部修改原张量影响模型。

### 2.9 自动求导和反向传播

最小训练示例：

```python
optimizer.zero_grad()
logits = model(images)
loss = F.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```

含义：

1. `zero_grad()`：清空上一次梯度；
2. `model(images)`：前向计算；
3. `cross_entropy`：预测与标签的误差；
4. `backward()`：从 loss 反向计算所有参数梯度；
5. `step()`：根据梯度更新参数。

V7 训练阶段就是这个循环的扩展版。

### 2.10 `train()` 和 `eval()`

```python
model.train()
model.eval()
```

二者主要影响 Dropout、BatchNorm 等训练行为。V7 训练源码同步类别原型时会临时切到 `eval()`，原因是不能把一次随机 Dropout 结果永久写入导出原型。

### 2.11 `torch.no_grad()`

```python
@torch.no_grad()
def v7_logits(...):
    ...
```

部署和评估只需要预测，不需要梯度。关闭梯度能减少显存占用并避免意外训练。

---

## 3. GZSL、CLIP 和类别原型

### 3.1 Seen 与 Unseen

在 CUB 数据集中共有 200 个鸟类：

- 150 个 seen 类：训练时有图片和标签；
- 50 个 unseen 类：训练时没有图片，但允许使用类别文本语义；
- 测试时，seen 和 unseen 图片都会出现。

普通分类只在训练见过的类别中选择。GZSL 更困难，因为测试时必须让 seen 和 unseen 共同竞争。

### 3.2 图片如何变成 768 维向量

CLIP 图像编码器把图片转换成语义向量：

```text
原始图片 → CLIP ViT-L/14@336 → CLS feature x∈R⁷⁶⁸
```

可以把 768 个数字想象成 768 个抽象视觉坐标。它们不是手工定义的“红色、长嘴”等属性，而是 CLIP 从海量图文数据中学习出的表示。

### 3.3 类别原型是什么

一个类别原型也是 768 维向量。例如：

```text
P[0] = 类别0的语义原型
P[1] = 类别1的语义原型
...
P[199] = 类别199的语义原型
```

图片分类：

```text
logit_c = normalize(x) · P[c]
```

200 个 `logit_c` 中最大的类别就是预测结果。

### 3.4 U、S、H、ZS

- `U`：在 200 类共同竞争时，unseen 类图片的平均每类准确率。
- `S`：在 200 类共同竞争时，seen 类图片的平均每类准确率。
- `H`：U 与 S 的调和平均，要求两边同时好。
- `ZS`：只让 50 个 unseen 类竞争时的准确率。

调和平均：

```text
H = 2US / (U+S)
```

V7：

```text
U = 77.606910
S = 83.639657
H = 2×77.606910×83.639657/(77.606910+83.639657)
  = 80.510432
```

为什么不只看普通平均？因为如果模型只会 seen 类，例如 `S=95、U=5`，普通平均还有 50，但调和平均只有约 9.5，会强烈惩罚失衡。

---

## 4. 为什么需要“类别关系图”

### 4.1 单类别描述的问题

很多鸟类描述会重复：

```text
small bird
brown wing
white belly
```

真正决定类别的经常是相对差异：

```text
A 的喙更细，而 B 的喙更粗
A 有白色眼圈，而 B 没有
```

因此 V7 不只保存“类别 A 是什么”，还保存“A 相对相似类别 B 有什么不同”。

### 4.2 图中的节点和边

- 节点：200 个类别；
- 边：两个相似类别之间存在比较关系；
- 当前共有 438 条无向类别边；
- 每条边有两个文本方向：`A rather than B` 和 `B rather than A`。

关系文本编码为：

```text
relation_embeddings.shape = [438,2,768]
```

有向差异：

```text
D[e] = text(A rather than B) - text(B rather than A)
```

所以：

```text
D.shape = [438,768]
```

---

## 5. Incidence 矩阵：用矩阵记录边方向

Incidence matrix 可译为“关联矩阵”。它记录每条边从哪个节点出发、到哪个节点结束。

假设只有三个类别 A、B、C，两条边：

```text
e₁: A → B
e₂: B → C
```

规定边起点为 `+1`，终点为 `-1`：

```text
        A   B   C
e₁     +1  -1   0
e₂      0  +1  -1
```

因此：

```text
B = [[ 1,-1, 0],
     [ 0, 1,-1]]
```

V7 真实代码：

```python
incidence = torch.zeros(EDGE_COUNT, CLASS_COUNT, dtype=torch.float64)
rows = torch.arange(EDGE_COUNT)
incidence[rows, edges[:, 0]] = 1.0
incidence[rows, edges[:, 1]] = -1.0
```

真实 shape：

```text
B.shape = [438,200]
```

### 5.1 `B @ G` 的意义

假设每个类别有一个向量 `G[c]`。对于边 `A→B`：

```text
(B @ G)[e] = G[A] - G[B]
```

所以 `B @ G` 给出的正是“每条边两端类别原型的差”。

这就是整个图编译问题的关键：

> 找到 200 个类别向量 G，使它们在每条边上的差尽可能符合文本差异 D。

---

## 6. 图拉普拉斯 `L=BᵀB`

### 6.1 三类别例子

刚才：

```text
B = [[ 1,-1, 0],
     [ 0, 1,-1]]
```

计算：

```text
L = BᵀB
  = [[ 1,-1, 0],
     [-1, 2,-1],
     [ 0,-1, 1]]
```

观察：

- 对角线是节点的度数，也就是连接多少条边；
- 非对角线的 `-1` 表示两个节点直接相连；
- 每行和为 0。

拉普拉斯的作用可以理解为：用一个矩阵同时表达“谁与谁相连”和“相邻节点应该如何协调”。

### 6.2 为什么叫 Laplacian

连续空间的拉普拉斯算子衡量一个点与周围的差异；图拉普拉斯做类似的事，只是“周围”变成图上的邻居。

对于节点数值向量 `g`：

```text
(Lg)[i] = degree(i)×g[i] - Σ邻居j g[j]
```

如果某节点与邻居非常接近，这个值较小；如果它与邻居差异很大，这个值较大。

---

## 7. 从边文本 D 求类别原型 G

### 7.1 第一性原理目标

我们希望：

```text
B G ≈ D
```

左边 `BG` 是类别原型在每条边上的差；右边 `D` 是文本告诉我们的边差异。

因为文本关系可能存在噪声或循环矛盾，通常无法严格相等，所以最小化平方误差：

```text
min_G ||BG-D||²_F
```

再加入正则项，防止解过大或矩阵不可逆：

```text
min_G ||BG-D||²_F + λ||G||²_F
```

这里：

- `||·||²_F`：把矩阵中所有元素平方后求和；
- `λ`：ridge 正则强度；V7 使用 `λ=0.3`。

### 7.2 手工求导

对 G 求导：

```text
∂/∂G [||BG-D||² + λ||G||²]
= 2Bᵀ(BG-D) + 2λG
```

最优点的导数为 0：

```text
Bᵀ(BG-D) + λG = 0
BᵀBG - BᵀD + λG = 0
(BᵀB + λI)G = BᵀD
```

所以：

```text
G = (BᵀB + λI)⁻¹ BᵀD
```

令：

```text
M = (BᵀB + λI)⁻¹ Bᵀ
```

得到：

```text
G = MD
```

### 7.3 为什么代码不用 `inverse`

数学上写逆矩阵，代码使用：

```python
mapping = torch.linalg.solve(system, incidence.T)
```

对应：

```text
system × mapping = Bᵀ
```

其中：

```text
system = BᵀB + λI
```

`solve` 比先算逆矩阵再乘法更稳定、更准确。

### 7.4 `λI` 解决什么问题

图拉普拉斯有一个天然问题：如果给所有节点同时加同一个常数，边差不会变化。

```text
(G[A]+c) - (G[B]+c) = G[A]-G[B]
```

因此没有正则时，解可能不唯一。加上 `λI` 后：

- 方程变成稳定可解；
- 限制类别关系原型过大；
- 即使图有多个连通分量也能得到确定解。

### 7.5 为什么同时翻转边方向不改变结果

如果一条边从 `A→B` 改成 `B→A`：

```text
B[e] → -B[e]
D[e] → -D[e]
```

两者同时变号后：

```text
BᵀB 不变
BᵀD 不变
G 不变
```

Gate A 实测方向翻转误差为 0，说明实现没有偷偷依赖任意边方向编号。

---

## 8. 为什么边打分可以编译成类别打分

Reader 输出图片关系向量：

```text
u = Reader(x) ∈ R⁷⁶⁸
```

逐边打分：

```text
r = uDᵀ
```

Shape：

```text
u  : [B,768]
Dᵀ : [768,438]
r  : [B,438]
```

再从边映射到节点：

```text
node_score = rMᵀ
           = uDᵀMᵀ
```

由于：

```text
G = MD
Gᵀ = DᵀMᵀ
```

所以：

```text
uDᵀMᵀ = uGᵀ
```

右边只需要 200 个类别关系原型 G，不再需要逐图片生成 438 个边分数或执行拉普拉斯求解。

Gate A 的实测最大误差：

```text
4.218847×10⁻¹⁵
```

这接近浮点数计算误差，证明线性部分确实可以编译。

---

## 9. V7 的三个模块 S、V、I

## 9.1 S：Semantic Role Prototypes

为什么存在：单一类别名无法表达所有细粒度视觉差异。

每个类别有 8 个角色文本原型：

```text
R.shape = [200,8,768]
```

训练 8 个角色权重 `w_r`：

```text
Q_image = P + Σᵣ wᵣRᵣ
```

代码：

```python
role_residual = torch.einsum("r,crd->cd", self.role_weights(), self.role_q)
return self.base_q + role_residual
```

`einsum` 这里做的只是：把 8 个角色向量乘各自权重，再相加。

S-off：

```text
Q_image = P
```

V7 中关闭 S 后 H 降低 `1.350275`。

## 9.2 V：Visual Relation Reader

为什么存在：原始 CLIP 图片向量适合图文整体匹配，但未必直接擅长读取“类别 A 相对 B 的差异方向”。

公式：

```text
u(x)=normalize(x + W₂ GELU(W₁x))
```

Shape：

```text
x       : [B,768]
W₁      : [64,768]
W₁x     : [B,64]
W₂      : [768,64]
W₂(...) : [B,768]
u       : [B,768]
```

为什么先压到 64 维再回到 768 维：

- 参数更少；
- 限制模型容量，避免记住所有训练图片；
- 逼迫 Reader 学习一个共享的关系读取规则。

为什么有残差 `x + ...`：

- 初始状态可以接近原始 CLIP 特征；
- Reader 只需学习修正量；
- 训练不稳定时仍保留原始视觉信息。

为什么使用 GELU：它是平滑非线性函数，使 Reader 能表达比单一线性变换更复杂的规则。

V-off：

```text
u = normalize(x)
```

即保留关系类别原型，但关闭学习出来的视觉修正。V7 中 V-off 使 H 降低 `1.087737`。

## 9.3 I：Incidence Relation Compiler

为什么存在：单条比较关系只说明两个类别，最终分类却需要 200 个类别共同竞争。

I 将：

```text
438 条有向文本差异 D
→ incidence/Laplacian 图积分
→ 200 个类别关系原型 G
```

关系 logits：

```text
relation_logits = α × uGᵀ
```

`α` 是整体关系强度。代码用 sigmoid 把它限制在 `(0, alpha_max)`：

```python
def alpha(self):
    return self.alpha_max * torch.sigmoid(self.raw_alpha)
```

这样训练不会把关系修正无限放大。

I-off：

```text
α = 0
```

V7 中关闭 I 后 H 降低 `1.313951`。

---

## 10. 最终分类矩阵 Q 是怎么组成的

图片端：

```text
h(x) = concat(normalize(x), Reader(x))
```

所以：

```text
h.shape = [B,1536]
```

类别端：

```text
Q = concat(Q_image, αG)
```

所以：

```text
Q.shape = [200,1536]
```

点积展开：

```text
hQᵀ
= [normalize(x),u] · [Q_image,αG]ᵀ
= normalize(x)Q_imageᵀ + αuGᵀ
```

这正好等于：

- 基础原型与角色语义分数；
- 编译后的视觉关系分数。

最后加类别 bias：

```text
logits = hQᵀ + b
```

`b` 中的 seen 类位置保存固定校准值，unseen 类位置为 0。

---

## 11. 训练阶段代码地图

正式 V7 部署代码在 `model/frameworks/v7/`。它的训练来源保留在已审实验实现 `model/frameworks/v6/`，因为正式 Tag 不重写已经发生的训练历史。

### 11.1 构造图编译原型

文件：`model/frameworks/v6/compiled_pclr.py`

关键位置：

- `CompiledPCLRHead`：约第 46 行；
- incidence 构造：约第 111 行；
- `system=BᵀB+λI`：约第 115 行；
- `torch.linalg.solve`：约第 118 行；
- `compiled_g=M@D/temperature`：约第 120 行。

核心代码：

```python
incidence = torch.zeros(EDGE_COUNT, CLASS_COUNT, dtype=torch.float64)
incidence[rows, edges[:, 0]] = 1.0
incidence[rows, edges[:, 1]] = -1.0

system = incidence.T @ incidence + ridge_lambda * torch.eye(CLASS_COUNT)
mapping = torch.linalg.solve(system, incidence.T)
direction = relation_embeddings[:, 0] - relation_embeddings[:, 1]
compiled_g = mapping @ direction / relation_temperature
```

使用 float64 求解图系统，是为了降低矩阵求解误差；生成 G 后再转成 float32，用于普通神经网络计算。

### 11.2 Reader

关键位置：`read_images`，约第 251 行。

```python
residual = self.reader_out(F.gelu(self.reader_in(values)))
return F.normalize(values + residual, dim=-1)
```

### 11.3 角色语义

关键位置：`image_q`，约第 258 行。

```python
role_residual = torch.einsum("r,crd->cd", self.role_weights(), self.role_q)
return self.base_q + role_residual
```

### 11.4 关系方向 CE

关键位置：`relation_direction_loss`，约第 297 行。

对于真类 y，只选择包含 y 的 seen-seen 边：

```text
E_y = {e | y 是边 e 的一个端点}
```

Reader 分别与两个方向文本做点积：

```python
scores = torch.einsum("bd,ekd->bek", readout, relations)
```

Shape：

```text
readout   : [B,768]
relations : [438,2,768]
scores    : [B,438,2]
```

每条有效边是一个二分类问题：当前图片更符合方向 0 还是方向 1。

不包含真类的边必须忽略，因为一张类别 A 的图片不能为“B 相对 C”提供可靠监督。

### 11.5 最终 200 类 CE

关键位置：`training_losses`，约第 328 行。

```python
logits = self(images)
classification = F.cross_entropy(logits, targets)
relation = self.relation_direction_loss(images, targets)
total = classification + relation_loss_weight * relation
```

最终分类 CE 直接训练：

- Reader；
- α；
- 8 个角色融合权重。

方向 CE 主要训练 Reader 如何读取关系方向。

### 11.6 为什么训练 TG/GTD

文件：`model/frameworks/v6/train_compiled_pclr.py`

- `_parent_loss`：约第 229 行；
- `_online_control_losses`：约第 253 行；
- `run`：约第 608 行。

TG/GTD 使用：

```text
Parent CE + topology loss + GTD gate loss
```

V7 训练共 200 名义 epoch、28,228 次更新。每一步重新随机排列 7,057 张 seen 训练图片并取前 50 张：

```python
indices = torch.randperm(7057, generator=generator)[:50]
```

unseen 图片不参与梯度。

---

## 12. 导出阶段

训练模型仍保存：

- 关系文本；
- edge index；
- compiled G；
- 各训练参数；
- S/V/I 关闭所需状态。

但部署不需要全部内容。`export()` 只输出：

```text
q
bias
reader_in_weight
reader_in_bias
reader_out_weight
reader_out_bias
```

这六个张量就是 V7 部署的全部计算状态。

V7 测试明确检查部署 `state_dict` 中不存在：

```text
edge_index
relation_embeddings
incidence
laplacian_map
```

---

## 13. 正式 V7 部署代码逐段解释

文件：`model/frameworks/v7/model.py`

### 13.1 固定维度

```python
CLASS_COUNT = 200
EMBED_DIM = 768
HIDDEN_DIM = 64
```

这些数字来自正式 CUB 和 Reader 合同。错误 shape 会直接报错，避免悄悄广播或错位。

### 13.2 加载六个导出张量

第 20—50 行：

```python
tensors = {
    "q": ...,
    "bias": ...,
    "reader_in_weight": ...,
    ...
}
```

接着检查：

```python
if tuple(value.shape) != expected[name] or not torch.isfinite(value).all():
    raise ValueError(...)
```

`torch.isfinite` 确保没有 NaN 或无穷大。

### 13.3 Reader 前向

第 66—77 行：

```python
hidden = F.linear(values, self.reader_in_weight, self.reader_in_bias)
residual = F.linear(F.gelu(hidden), self.reader_out_weight, self.reader_out_bias)
return F.normalize(values + residual, dim=-1)
```

`F.linear(x,W,b)` 等价于：

```text
xWᵀ+b
```

### 13.4 最终 logits

第 79—85 行：

```python
image = F.normalize(image_features.float(), dim=-1)
readout = self.read_images(image_features)
logits = torch.cat((image, readout), dim=1) @ self.q.T + self.bias
```

这里没有任何隐藏图运算。

### 13.5 checkpoint 身份检查

第 88—105 行会检查：

- Experiment 必须是 `V6-TRY-006`；
- RUN commit 必须正确；
- 训练 config SHA 必须正确；
- 必须存在 export 字段。

这样可以防止误加载别的实验 checkpoint 后仍输出看似合理的结果。

---

## 14. 正式 evaluator 逐段解释

文件：`model/frameworks/v7/evaluate.py`

### 14.1 配置和文件身份

第 27—44 行检查：

- V7 config schema；
- source RUN commit；
- source training config SHA；
- promotion source commit；
- checkpoint 文件 SHA；
- CUB 资产配置 SHA。

### 14.2 加载数据和模型

```python
asset_config, _ = load_config(asset_config_path)
tensors = load_assets(asset_config)
model, _ = load_v7_checkpoint(checkpoint_path)
```

注意：`load_assets` 只是读取固定特征和标签。真正 logits 只由 `V7DeploymentModel` 计算。

### 14.3 GZSL 预测

seen 测试图片和 unseen 测试图片都执行：

```python
prediction = logits.argmax(dim=1)
```

此时 `logits` 有 200 列，所以 U 和 S 都是 200 类共同竞争。

### 14.4 ZS 预测

```python
logits.index_select(1, unseen)
```

只保留 50 个 unseen 类列，再 argmax。这就是 ZS。

### 14.5 指标复现硬检查

```python
if abs(metrics[name] - expected) > 1e-6:
    raise RuntimeError(...)
```

正式 evaluator 不只是“算一个差不多的数字”，而是要求精确复现晋级结果。

---

## 15. 一张图片经过 V7 的完整过程

假设输入一张鸟图：

### 步骤 1：CLIP 特征

```text
x.shape = [1,768]
```

### 步骤 2：基础图片归一化

```text
x_norm = x / ||x||
```

### 步骤 3：Reader

```text
hidden   = W₁x+b₁        # [1,64]
residual = W₂GELU(hidden)+b₂  # [1,768]
u        = normalize(x+residual)
```

### 步骤 4：拼接

```text
h = [x_norm,u]
h.shape = [1,1536]
```

### 步骤 5：对每个类别评分

对于类别 c：

```text
logit[c] = h · Q[c] + b[c]
```

### 步骤 6：预测

```text
prediction = argmax(logits)
```

整个部署流程没有出现 438 条边，也没有调用 `solve`。

---

## 16. S/V/I 关闭实验为什么重要

完整模型性能高，不代表每个模块都有用。可能某模块只是摆设，或者被另一个模块完全替代。

因此在同一个 Full checkpoint 上分别关闭：

| 条件 | 关闭内容 | H | Full−off |
|---|---|---:|---:|
| Full | 全部开启 | 80.510432 | — |
| S-off | 去掉角色语义项 | 79.160157 | 1.350275 |
| V-off | Reader 改为恒等读取 | 79.422694 | 1.087737 |
| I-off | α=0，关闭关系残差 | 79.196481 | 1.313951 |

这里的“同 checkpoint”非常重要：不允许每关闭一个模块就重新训练，因为那会让其余模块重新补偿，回答的是另一个问题。

关闭实验能证明：当前 Full 模型依赖这个组件。它不能单独证明组件具有学术新颖性。

---

## 17. 训练、导出、部署三者不要混淆

| 阶段 | 有关系图吗 | 有梯度吗 | 主要输出 |
|---|---:|---:|---|
| 训练 | 有 | 有 | 学会 Reader、α、角色权重、TG/GTD |
| 导出 | 有一次性 G/Q 构造 | 无 | 六个冻结部署张量 |
| 部署 | 无 | 无 | 200 类 logits |

常见误解：

### “部署没有图，是不是关系信息丢了？”

不是。关系信息已经进入 `Q` 的后 768 维类别权重。

### “Q 是不是只有关系原型？”

不是。Q 的前半部分是基础与角色原型，后半部分是 `αG`。

### “Reader 就是分类器吗？”

不是。Reader 只把图片特征变成更适合读取关系方向的表示；真正分类仍由 `hQᵀ+b` 完成。

### “为什么正式训练代码还在 frameworks/v6？”

因为它是产生 V7 checkpoint 的准确历史实现。V7 正式部署代码独立放在 `frameworks/v7`，不复制和改写已经发生的 RUN 身份。

---

## 18. 如何运行和验证

### 18.1 本地部署测试

在仓库根目录：

```powershell
python -m pytest tests/frameworks/v7/test_v7_deployment.py -q
```

测试覆盖：

- `hQᵀ+b` 与手工计算一致；
- checkpoint 身份正确；
- 部署 state 不含图资产；
- 缺少导出字段时拒绝加载。

### 18.2 服务器正式评估

```bash
python -m model.frameworks.v7.evaluate \
  --config config/framework_v7.yaml \
  --device cuda:0
```

预期输出：

```json
{
  "H": 80.51043185404096,
  "S": 83.639657497406,
  "U": 77.60691046714783,
  "ZS": 88.4734034538269
}
```

### 18.3 在 Python 中加载模型

```python
import torch
from model.frameworks.v7 import load_v7_checkpoint

model, checkpoint = load_v7_checkpoint(
    "/path/to/V6-TRY-006/model_best.pth"
)
model = model.cuda().eval()

image_features = torch.randn(4, 768, device="cuda")
with torch.no_grad():
    logits = model(image_features)
    predictions = logits.argmax(dim=1)

print(logits.shape)       # torch.Size([4, 200])
print(predictions.shape)  # torch.Size([4])
```

这里的随机特征只能测试代码 shape，不能产生有意义的鸟类预测。真实预测必须使用同一个 CLIP checkpoint和预处理生成的特征。

---

## 19. 常见代码错误

### 19.1 Shape 对不上

矩阵乘法要求中间维度一致：

```text
[B,1536] @ [1536,200] → [B,200]
```

如果 Q 错写成 `[1536,200]` 又调用 `Q.T`，就会失败。

### 19.2 忘记归一化

未归一化时，向量长度可能压过方向相似度，破坏训练与正式部署一致性。

### 19.3 训练态同步原型

TG 模块包含 Dropout。训练态直接读取原型，会把一次随机采样固化。源码同步前临时 `eval()`，并保存和恢复 RNG 状态。

### 19.4 用 `inverse` 替代 `solve`

显式求逆更慢、更不稳定。应保持：

```python
torch.linalg.solve(B.T @ B + lambda_ * I, B.T)
```

### 19.5 把 unseen 图片用于梯度

V7 允许使用 unseen 类文本语义，但不允许 unseen 图片或其标签参与训练梯度。

### 19.6 把 best-ZS 拼给 best-H

best-H 和 best-ZS 可能来自不同 checkpoint。正式 U/S/H/ZS 必须全部取 best-H 的同一个 checkpoint；best-ZS只能独立观察。

### 19.7 把模块贡献当成新颖性

S/V/I off 都超过 1 H，只能证明模型依赖它们。是否具有论文新意，还必须比较 PC-CLIP、Comparative Descriptors、DGP/GCN、TaskRes、HodgeRank 简化版和随机图对照。

---

## 20. 你应该掌握的数学知识清单

按学习顺序：

1. 向量、长度和点积；
2. 矩阵 shape 与矩阵乘法；
3. 转置 `Aᵀ`；
4. 导数、梯度和链式法则；
5. 最小二乘；
6. L2/ridge 正则；
7. 图的节点、边、度数；
8. incidence 矩阵；
9. 图拉普拉斯 `L=BᵀB`；
10. 线性方程 `AX=Y` 与 `solve`；
11. softmax、交叉熵；
12. 调和平均。

不需要先学完整图神经网络。V7 的核心图数学是固定线性方程，不是多层 message-passing GNN。

---

## 21. 推荐练习

### 练习 1：Shape 推导

给定：

```text
x:[50,768]
Q:[200,1536]
```

写出 Reader 输出、h、Q.T 和 logits 的 shape。

答案：

```text
Reader(x):[50,768]
h:[50,1536]
Q.T:[1536,200]
logits:[50,200]
```

### 练习 2：三节点 incidence

为边 `A→C`、`B→C` 写出 incidence 矩阵。

答案：

```text
B = [[1,0,-1],
     [0,1,-1]]
```

### 练习 3：为什么 `BᵀB` 不依赖边方向

如果某行 `b` 变成 `-b`：

```text
(-b)ᵀ(-b)=bᵀb
```

所以拉普拉斯不变。

### 练习 4：拆开最终点积

证明：

```text
[x,u][Q_image,αG]ᵀ = xQ_imageᵀ + αuGᵀ
```

这就是 V7 能把两个分支合并成一个矩阵乘法的原因。

### 练习 5：手动算 H

取 `U=60、S=90`：

```text
H=2×60×90/(60+90)=72
```

H 明显小于普通平均 75，因为它更惩罚 U/S 不平衡。

---

## 22. 代码阅读顺序

第一次建议按下面顺序：

1. `model/frameworks/v7/model.py`：只看最终部署，最短也最清楚；
2. `tests/frameworks/v7/test_v7_deployment.py`：看代码想保证什么；
3. `config/framework_v7.yaml`：看固定数字和身份；
4. `experiments/v7/framework_diagram.html`：看整体信息流；
5. `model/frameworks/v6/compiled_pclr.py`：看 S/V/I 如何训练和导出；
6. `model/frameworks/v6/train_compiled_pclr.py`：最后看完整训练循环；
7. `model/frameworks/v7/evaluate.py`：理解正式指标如何复现。

---

## 23. 最后用一张文字图总结

```text
                       训练阶段

类别角色文本 R ───────────────┐
                              ├─ S：角色语义 → Q_image ─────┐
TG + GTD 类别原型 P ──────────┘                              │
                                                             ├─ Q:[200,1536]
成对有向关系文本 D ─→ B/Laplacian ─→ G:[200,768] ─→ αG ────┘
                                        ▲
图片 CLIP 特征 x ─→ V：Reader ─→ u ─────┘

                       部署阶段

图片特征 x ─→ normalize(x) ─┐
                             ├─ h:[B,1536] ─→ hQᵀ+b ─→ 200类logits
图片特征 x ─→ Reader(x) ─────┘

部署阶段没有：Top-K、438边打分、Laplacian solve、cap、std、late ensemble。
```

如果你能从头解释这张图，并能说出每个张量的 shape，就已经掌握了 V7 的主体。
