# V5-TRY-004 OREF 冻结代码双Agent交叉审查

review_date: 2026-09-01
idea_id: IDEA-190
initial_code_commit: `7e7a0e5377622ff7978272d9f48ba53dfc54b16d`
fix_commits: [`06388b1a8aee4d1b38211176f8619a7e9a647232`, `a2e063a15008204c1da5f803478c88faef145a69`]
final_reviewed_code_commit: `a2e063a15008204c1da5f803478c88faef145a69`
review_agents: [`/root/rceg_code_a`, `/root/rceg_code_b`]

## 冻结范围与最小证据

- `model/frameworks/v5/oref.py`
- `model/frameworks/v5/oref_data.py`
- `model/frameworks/v5/train_oref.py`
- `model/frameworks/v5/evaluate_oref_dev.py`
- `tools/prepare_oref_assets.py`
- `tests/frameworks/v5/test_oref.py`
- 初始专项`7 passed`；集中修复后最终专项`10 passed`。

## 独立初审与直接交叉

- Agent A：`P0=0 / P1=2 / P2=5 / revise`。阻断项为Target-free逐类receipt未绑定class ID/150候选轴，以及eval未强制checkpoint training commit等于当前运行commit。
- Agent B：`P0=0 / P1=1 / P2=4 / revise`。阻断项同为Target-free class-order/150轴不完整。
- 双方直接交换完整清单后共同确认两个P1：Target-free必须绑定bundle、rows、active axis、per-class IDs和metric；所有checkpoint必须由同一expected commit训练。

## 集中修复与第一次复核

- `06388b1`补齐Target-free按class ID重排、150/50数量门、checkpoint expected commit、off opened keys/call counts、witness IDs、manifest class IDs、output晚建与`preliminary_gate_passed`。
- 两名Agent复核后继续发现：Target-free receipt只与config轴自洽，尚未强制等于OREF实际`full.class_ids`。
- 双方直接交叉维持`P0=0 / P1=1 / revise`。

## 最终修复与通过

- `a2e063a`强制`config.targetfree_active_class_ids == full.class_ids`，并将当前OREF轴传入receipt对齐；labels必须属于active轴。
- checkpoint进一步校验train schema、condition、score mode、training commit、bundle和两步梯度receipt；eval配置使用精确key集合。
- 最终双方独立复核均为`P0=0 / P1=0 / pass`，再直接交换终审清单并确认无新P0/P1。
- 最终共同结论：**双Agent交叉审查通过**。

非阻断P2：eval配置中的`asset_generation_commit`字段在当前代码未单独比较bundle commit；checkpoint拒绝路径可继续增加负例测试。真实配置仍必须把该字段绑定manifest记录的生成commit。

该签字仅覆盖冻结代码。资产manifest、Target-free逐类收据、训练/eval config SHA、服务器环境/GPU和最终RUN commit必须在运行前后继续绑定；任何module/forward/loss/数据/评估语义变化使签字失效。
