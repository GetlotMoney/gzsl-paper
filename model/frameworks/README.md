# 正式框架代码

这里只保存 owner 已接纳的模型身份。每个版本的模型与训练入口自包含；正式框架不得依赖 `model/candidates/`。

| 版本 | 方法 | 模型入口 | 训练入口 |
|---|---|---|---|
| V1 | GTPJ | `model.frameworks.v1.model.GTPJ` | `python -m model.frameworks.v1.train` |
| V2 | TG-VPR-H1 | `model.frameworks.v2.model.TGVPRH1FixedEqual` | `python -m model.frameworks.v2.train` |
| V4 | TG+GTD | `model.frameworks.v4.gtd.GTDTSTModel` | `python -m model.frameworks.v4.train`（仅复现晋级来源RUN） |

V3 是已关闭的探索阶段，没有伪造正式 `framework/v3` 或 Tag `v3`。V4 为保持正式 checkpoint 的完整 `state_dict`，同时保留 TG、TST、CCGR 父模型结构与 GTD。当前V4训练器仍校验V3晋级RUN身份，新的V4训练配置尚未建立。

正式checkpoint均为`state_dict`或checkpoint字典；除V1的`model.MyModel.GTPJ`外，不承诺兼容旧模块路径下的完整Python对象pickle。
