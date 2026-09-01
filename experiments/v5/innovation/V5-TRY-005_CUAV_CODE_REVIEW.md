# V5-TRY-005 CUAV 冻结代码双Agent交叉审查

review_date: 2026-09-01
idea_id: IDEA-191
initial_code_commit: `e107f3c226c89a46fcf62b9ba3651fc1635e2b0e`
fix_commit: `36b21fda2a8f104c694fefa735ae2e6994fd8e17`
final_reviewed_code_commit: `36b21fda2a8f104c694fefa735ae2e6994fd8e17`
review_agents: [`/root/rceg_code_a`, `/root/rceg_code_b`]

## 冻结范围与证据

- `model/frameworks/v5/cuav.py`
- `model/frameworks/v5/cuav_data.py`
- `model/frameworks/v5/train_cuav.py`
- `model/frameworks/v5/evaluate_cuav_dev.py`
- `tools/prepare_cuav_assets.py`
- `tests/frameworks/v5/test_cuav.py`
- 初始专项`6 passed`；集中修复后专项`7 passed`。

## 独立初审与直接交叉

- Agent A：`P0=0 / P1=3 / P2=4 / revise`。阻断项为S-off先算歧义再清零、V-off预加载all25 low-res cache、StaticBest train manifest未同bundle。
- Agent B：`P0=0 / P1=2 / P2=5 / revise`。阻断项为train/oracle未绑定同bundle和crop geometry/window未强校验。
- 双方交换完整清单后共同合并为：S-off物理关闭、V-off真实B=1 selected low-res forward、train/eval/oracle同bundle、固定25窗口/boxes同身份四类P1。

## 集中修复与复核

- S-off在Parent logits后早返回，不调用top2/query/stats；测试用monkeypatch证明ambiguity未打开。
- V-off删除all25 low-res feature cache，改存`preprocessed_336`，action确定后只裁选中窗口并执行一次CLIP；测试记录selected rows count。
- eval分别validate `dev_train/dev_eval/dev_eval_oracle` subset SHA和bundle；强制4702/2355、100/150/50轴与labels membership。
- loader校验`crop_action_sha256`和全部tensor shape；Full raw selected boxes逐值等于asset boxes。
- checkpoint identity和B=1 raw-open/forward收据补齐。
- 两名Agent独立复核均为`P0=0 / P1=0 / pass`，再直接交换终审清单并确认无新P0/P1。

最终共同结论：**双Agent交叉审查通过**。

非阻断P2仅为常量复用、opened asset keys、StaticBest/action SHA和reproducibility flags等留痕增强，不改变当前方法、指标或B=1语义。

本签字仅覆盖冻结代码。资产manifest、训练/eval config SHA、policy checkpoint、服务器环境/GPU和最终RUN commit必须在真实运行前后继续绑定；任何module/forward/loss/数据/crop geometry/评估语义变化使签字失效。
