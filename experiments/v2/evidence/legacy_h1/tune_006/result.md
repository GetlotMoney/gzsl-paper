# V5-TUNE-006 最终结果

## 最终决定

TG-VPR-H1正式冻结为：三组固定等权、单一768维Value路径、`inner_ratio=0.35`、`outer_ratio=0.65`、`topology_weight=0.1`。不保留可学习组权重，不再搜索head、inner/outer residual、topology或组权重，也不进入方案三、方案四。

## 四seed结果

| seed | 来源 | U | S | H | ZS |
|---:|---|---:|---:|---:|---:|
| 5 | RUN-009 | 73.333913 | 74.088860 | 73.709453 | 81.534684 |
| 6 | RUN-010 | 73.258799 | 74.515074 | 73.881597 | 81.534684 |
| 7 | RUN-004 / V5-ABLATION-014 RUN-003 | 72.655779 | 75.443041 | 74.023182 | 81.534684 |
| 8 | RUN-011 | 73.360914 | 74.240613 | 73.798142 | 81.534684 |

统计：

- U：mean=`73.152351`，min=`72.655779`，max=`73.360914`，range=`0.705135`
- S：mean=`74.571897`，min=`74.088860`，max=`75.443041`，range=`1.354181`
- H：mean=`73.853094`，min=`73.709453`，max=`74.023182`，range=`0.313729`
- ZS：mean/min/max=`81.534684`，range=`0`

与旧可学习权重H1对比，固定等权的四seed平均H提高`0.014826`；seed 5持平，seed 6/7/8分别提高约`0.016957/0.003518/0.038830`。提升很小，不作为新性能claim，但足以证明删除可学习权重没有代价。

## 第一阶段机制结论

第一阶段没有挑战者。关闭topology时，inner越大，U越高而S越低，H持续下降；topology对H的补偿从inner 0.20时的`+0.551924`增大到inner 0.65时的`+2.160581`。因此：

1. inner residual控制Value重参数化强度。
2. topology loss约束类别几何漂移，Value改写越强越需要它。
3. 当前`0.35/0.1`是粗网格中的最佳平衡点，而不是任意拍定。

## 证据边界

本实验共新增10个真实服务器RUN：第一阶段7个、第二阶段3个；低于13个硬上限。所有训练位于`lab4090`物理GPU 1并严格串行，0失败。结果使用official test做收口筛选，属于`not_confirmation_evidence`，不能冒充独立confirmation或promotion证据。

- 第一阶段提交：`4cf8baafdf9022c1e3b900a528d849fd831bc945`
- 第二阶段提交：`4c5c0910b7b498ed6e4021bc76a9a7236a8ce15e`
- Warehouse：`server:/data/lby/projects/cv_project/GTPJ_Warehouse/runs/v5/tune/V5-TUNE-006-closeout/`
