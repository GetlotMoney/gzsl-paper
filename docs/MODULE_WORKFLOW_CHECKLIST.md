# 新模块工作流检查清单

本文件是所有新模块从“准备做”到“通过、拒绝或替换”的唯一操作清单。具体事实仍分别写入Idea卡、框架实验队列和仓库外输出目录，不在本文件复制实验结果。

## 一、模块准备

- [ ] owner已明确本候选的准确父commit；新候选从该commit创建独立`exp/vX/<kind>/<module>`分支，不继承上一失败候选的模型代码。
- [ ] 分配新的`IDEA-xxx`，不得复用旧编号。
- [ ] 在`research/ideas/IDEA-xxx_<slug>.md`写明问题、证据、可证伪假设、唯一核心改动、父条件和失败条件。
- [ ] 在`research/IDEA_TREE.md`登记模块节点和当前状态：`proposed / testing / supported / revised / rejected`。
- [ ] 记录准确base commit、父条件RUN、父条件U/S/H/ZS、数据资产ID与manifest SHA。
- [ ] 说明模块输入、输出、公式、loss、trainable参数和关闭行为。
- [ ] 模块关闭时必须严格返回父模型；不能一次关闭两个实验模块冒充单模块消融。
- [ ] 多个逻辑不可拆分的组件若作为一个模块，必须统一命名、统一训练、统一关闭和统一计算贡献。
- [ ] 不使用人工属性、部位标注、框或专家残差；标准seen类别标签允许用于监督训练。

## 二、计算与语义检查

- [ ] 估算新增参数量、训练时间复杂度、评估复杂度、缓存体积和峰值GPU显存。
- [ ] 视觉模块必须先做真实batch50前向/反向smoke并记录视觉梯度和峰值显存。
- [ ] 纯运行优化不得改变输入信息、forward、loss或选模语义；发生语义变化时必须作为新Idea和新Experiment。
- [ ] 模块拥有TG父模型没有的独立信息或独立训练目标；不能只堆叠同类原型残差。
- [ ] 修改module、forward、loss、数据流或输入输出时，新增或更新对应`framework_diagram.html`。

### 视觉模块精度优先专项

- [ ] 首个视觉证据对照必须保留完整`24×24=576`个最终patch；不得只为缩短运行时间先压缩或硬删除token，再把精度下降归因于视觉信息无效。
- [ ] 多尺度候选固定说明`24×24 / 12×12 / 6×6`各自负责的细节、部位和区域证据，并分别保留完整576对照。
- [ ] 主体定位默认使用无人工框/部位的软权重，保留全部patch；任何硬选择必须报告被删除信息和相同计算预算对照。
- [ ] 局部文本优先使用可对应具体区域的前六角色；整体/独特描述进入局部分支时必须证明不重复TG全局证据。
- [ ] TG负责200类候选召回，视觉分支只在预注册Top-K内做类别两两差异判别；必须报告Top-K召回上限、修正数、破坏数和真类不在候选集的比例。
- [ ] 视觉证据必须有可靠性/拒绝机制；低置信、角色不一致或尺度不一致时严格保持TG排序，不能强制改动所有样本。
- [ ] 任何patch选择/去噪方案必须与“完整576、不选择”条件比较；精度优先于时间复杂度，只有不降低预注册精度门槛后才讨论加速。
- [ ] 使用LaSt-ViT类频域方法时，必须明确它是沿每个patch的特征通道轴做1D FFT，不得描述为空间网格频域；频率打分使用最终block的投影前hidden token，文本匹配使用`ln_post + visual.proj + L2`后的patch，并按位置对齐。
- [ ] 频域选择必须预注册paper公式与repo公式、Gaussian参数、每通道K、vote-count转patch权重方式及是否训练；不得把“每通道选择K个patch”误写成全局只保留K个空间patch。
- [ ] LaSt-ViT原论文依赖带选择聚合的预训练/微调；把其分数后处理到冻结CLIP只能作为新Idea的无训练诊断，不能直接复用论文提升claim。

## 三、运行前登记

- [ ] 当前分支保留此前全部版本实验账本；旧`experiments/v*/`目录只读，不删除、不移动、不重编号。
- [ ] 新TRY只登记到当前版本`experiments/vX/EXPERIMENT_QUEUE.csv`，跨版本父证据引用原RUN而不复制。
- [ ] 当前只使用一段式端到端联合训练，不再启动三段式、五段式或stage-best handoff候选。
- [ ] V3探索筛选固定CUB、seed7、batch50、每141步official评估，最多150名义epoch并按预注册里程碑动态停止；胜出累计条件和最终单模块移除才固定200名义epoch。
- [ ] 固定披露：`test_used_for_selection=true`、`unseen_images_used_for_gradient=false`、`strict_blind_claim=false`。
- [ ] 在当前框架`experiments/vX/EXPERIMENT_QUEUE.csv`新增一行`planned` TRY。
- [ ] TRY绑定准确config、唯一改动、seed、code commit和仓库外output URI。
- [ ] 本地相关测试和服务器相关测试通过，工作树clean后才能启动。
- [ ] 第一轮独立Agent已完成对抗审查；P0/P1已全部修复并有直接测试。
- [ ] 第二轮由另一独立Agent审查准确post-fix commit并明确“无P0/P1，第2轮通过”；此后代码未再变化。
- [ ] 冻结准确RUN commit/config；纯配置、队列、结果、文档不启动代码审核。
- [ ] 共享证据只生成一次：专项测试、一次完整测试、每objective一次GPU micro-batch、一次正常checkpoint恢复。
- [ ] Round 1只审公式/初始化/梯度/数据/Full-Off；Round 2只审身份/clean/真实评估/resume；两轮目标5–10分钟。
- [ ] 默认不做恶意checkpoint篡改、fuzz、逐张量hash链、重复完整测试、重复GPU或重复大资产SHA。
- [ ] P0/P1一次性汇总，只做一批集中修复；复核只审最终diff，未变化证据直接复用。
- [ ] 两次命令失败后停止自造入口，改用现有训练/测试入口；不新增receipt、证据页、状态机或额外目录。
- [ ] 第二轮明确“无P0/P1，第2轮通过”后立即启动，不增加第三轮准备。
- [ ] 生成后资产在生成代码、config和父身份不变时只做manifest/SHA/shape/dtype/count contract；身份变化则重新审核受影响范围。
- [ ] 不自动push GitHub，不移动正式framework分支或Tag。

## 四、硬门槛实验

- [ ] 使用准确父条件运行`Parent + Module`。
- [ ] 完整列出父条件与候选的U/S/H/ZS及`ΔU / ΔS / ΔH / ΔZS`。
- [ ] 计算累计加入贡献：`ΔH_add = H(Parent+Module) - H(Parent)`。
- [ ] 构建完整模型单模块移除条件，计算：`ΔH_remove = H(Full) - H(Full-Module)`。
- [ ] 首个新增模块若完整模型只有`Parent+Module`，其父条件同时就是`Full-Module`，一个候选RUN可同时回答加入和移除。
- [ ] `ΔH_add >= 1.000`且`ΔH_remove >= 1.000`才算模块通过。
- [ ] U/S差达到8点直接淘汰；5至8点标记风险但不能隐藏。
- [ ] U/S/H/ZS必须来自同一个best-H checkpoint，禁止跨checkpoint拼数字。

## 五、判定与止损

- [ ] `pass`：累计加入和单模块移除均至少+1 H，且U/S差未达到8点。
- [ ] `drop`：任一硬门槛失败，或出现严重偏置、饱和、非有限值、错误数据边界。
- [ ] 默认失败后立即换模块；只有owner当前明确批准时才进行有限单变量补救。
- [ ] 补救只改一个参数；不得通过多参数网格、换seed或修改成功条件制造通过。
- [ ] 模块失败后停止向其后继续堆新模块；先替换当前失败模块。
- [ ] 不为凑论文创新数量降低1点硬门槛。

## 六、结果回填

- [ ] 回填`EXPERIMENT_QUEUE.csv`：status、U/S/H/ZS、best epoch、decision、准确code commit和output URI。
- [ ] 回填Idea卡：真实结果、相对父条件增量、诊断、失败原因和最终状态。
- [ ] 更新`research/IDEA_TREE.md`中的模块状态。
- [ ] 失败RUN保留training.log、metrics.json、evaluation_history.json、model、config snapshot和数据指纹；不得删除或覆盖。
- [ ] 失败模块不创建正式Innovation目录；只保留Idea、TRY和仓库外证据。
- [ ] 只有`promote`候选才建立正式Experiment最小四文件和必要HTML图。
- [ ] owner明确接纳后，才能建立新的正式`framework/vY`与Tag `vY`。

## 七、每次启动前的最短复核

```text
Idea已登记？
父条件准确？
模块关闭严格返回父模型？
只使用一段式？
U/S/H/ZS和1点门槛已预注册？
batch50真实smoke和显存已验证？
config、commit、资产SHA、output URI齐全？
测试通过且本地/服务器clean？
两轮独立Agent已审查同一最终代码身份，且第二轮明确无P0/P1？
```

任一项为“否”，不得启动训练。

## 八、正式RUN运行闭环

- [ ] 启动后1分钟内确认真实Python子进程、预期GPU、准确HEAD、config SHA、checkpoint存在且`update>0`；不能只记录launcher PID。
- [ ] 结束时确认进程退出，且`metrics.json`、完整评估历史、`stop_reason=completed_fixed_150`、`total_updates=21171`、`history_length=152`全部成立。
- [ ] 双卡队列中，前一RUN完成后立即检查同卡下一RUN；输出目录不存在且GPU空闲时直接启动，并重复启动确认。
- [ ] 确认`loaded_training_checkpoints=[]`，RUN commit/config SHA/资产身份准确，U/S/H/ZS与Full/Off来自同一best-H checkpoint。
- [ ] 运行闭环只读取现有RUN证据，不新增Agent、测试、审核轮次、守护进程、状态机或文档层级。
