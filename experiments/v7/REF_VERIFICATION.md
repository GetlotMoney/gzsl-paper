# FRAMEWORK-V7正式身份与部署复现收据

- 核验日期：2026-09-02
- 正式commit：`b32a16f848c34f8e09d03b27d2f22ed445b9a295`
- 本地：`framework/v7 == v7^{} == b32a16f848c34f8e09d03b27d2f22ed445b9a295`
- 服务器：`HEAD == framework/v7 == v7^{} == b32a16f848c34f8e09d03b27d2f22ed445b9a295`
- 服务器worktree：`/data/lby/projects/cv_project/gzsl-paper-worktrees/framework-v7`
- 部署config：`config/framework_v7.yaml`，SHA256
  `7c806382b6d1899a3639ed16cd287c7894b210efda58707358172b2224b943dd`
- source checkpoint SHA256：`a551de9d182222141ab4be9db1ae2020417be3a7a7d1d4b369510d635f2207c9`
- evaluator：`python -m model.frameworks.v7.evaluate --config config/framework_v7.yaml --device cuda:0`
- 复现结果：`U/S/H/ZS=77.60691046714783/83.639657497406/80.51043185404096/88.4734034538269`
- 服务器正式worktree核验后保持clean。

结论：正式分支与Tag指向同一不可移动commit；独立V7部署入口只使用Reader、`Q`和`b`，已在
真实checkpoint和CUB资产上精确复现晋级指标。
