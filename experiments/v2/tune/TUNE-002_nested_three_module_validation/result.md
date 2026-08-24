# V2-TUNE-002 结果

状态：`rescue1_planned_lower_inner_weight`。

外层固定使用xlsa17标准100/50类别不相交validation。内层三折只覆盖外层100个训练类，pseudo-unseen类别数为`34/33/33`，外层50个validation类不进入任何梯度。

RUN-001唯一新增训练语义：每个主训练batch额外计算三个inner fold的平衡pseudo-seen/pseudo-unseen CE，三折取mean并以固定权重`1.0`加入总loss。official test不会被加载。

RUN-001在第一批训练前因共享原型接口仍限制support类只能为100/150而失败；内层折实际需要66/67类。该工程失败没有方法结果，修复接口守卫后以`RUN-001-RERUN`同条件重跑。

对照条件为`V2-TUNE-001/RUN-001`：`U/S/H/ZS_val=76.424742/76.521248/76.472964/79.934835%`。

RUN-001-RERUN完成50轮，最佳epoch=`3`，`U/S/H/ZS_val=70.510274/81.352049/75.544153/80.954689%`，相对旧外层baseline的H下降`0.928811`。随着训练继续，S升至约84而U降至约62，说明权重`1.0`的inner CE过强地推动seen判别，不能保留。

RESCUE-1只把`inner_episode_weight`从`1.0`降为`0.1`，fold、外层数据、三模块结构和学习率均保持不变。模型SHA256：`6ee3f7d71f9e765290459db6a95712d3d94f26c7f84074cd53be8bf5e1b34e56`；最后checkpoint SHA256相同。

RUN-002预注册为RESCUE-1：`inner_episode_weight=0.1`，其余配置逐项沿用RUN-001。

RUN-002最佳epoch=`16`，`U/S/H/ZS_val=72.324175/81.354070/76.573831/78.784090%`。H相对旧baseline提高`0.100867`，但U下降`4.100567`、ZS下降`1.150745`，属于seen提升驱动，不能认定为更强unseen泛化。

RESCUE-2保持inner权重`0.1`，只切断inner episode loss到TG-VPR的梯度，让inner pseudo-unseen监督仅更新TST-NTR与CCGR。模型SHA256：`cceffa925cfbfc9684b399067e81ef07d4e4143251e238f8791e9050fceee2d9`；最后checkpoint SHA256相同。

RUN-003预注册RESCUE-2：`inner_gradient_scope=transport_generator_only`；主训练CE和topology仍联合更新三个模块。

RUN-003最佳epoch=`16`，`U/S/H/ZS_val=73.057747/81.389427/76.998860/79.036897%`，H相对旧baseline提高`0.525896`，但U/ZS仍低`3.366995/0.897938`，不能仅凭H宣称unseen泛化增强。

RESCUE-3按episode-based ZSL的support/refining语义，将inner loss的图像batch改为pseudo-unseen-only；inner梯度仍只更新TST-NTR与CCGR，权重保持0.1。模型SHA256：`fc7a1f315e516380da98d5a57da9515b3819a6b6b82e34ed1aa8b6897608267f`；最后checkpoint SHA256相同。

RUN-004预注册RESCUE-3：每折inner batch固定64张pseudo-unseen图像，不再混入pseudo-seen图像。
