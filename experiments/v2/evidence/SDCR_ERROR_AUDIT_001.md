# SDCR_ERROR_AUDIT_001

source_model: `INNOVATION-041 / RUN-002 / seed 5`
model_sha256: `53f9065ddd5f32bc02ff4be3ce5db3c7a4eadf5117282b55a672780acec001ae`
output_uri: `/data/lby/projects/cv_project/GZSL_Warehouse/diagnostics/v2/sdcr_seed5_error_audit_20260824.json`
output_sha256: `7aa1e67e23d8de7132ea91cd91dfcc0cdab14c43ddd1694e8e8e64f568d3e58d`

复现指标：`U/S/H/ZS=76.747000/79.959893/78.320510/83.953977%`。

关键诊断：

- seen图像误预测为unseen的比例为`12.811792%`；unseen图像误预测为seen为`10.077519%`，两者接近，主瓶颈不是单向seen/unseen偏置。
- unseen在50类ZSL空间仍有固定困难类：Blue-winged Warbler仅`5%`，Baird Sparrow仅`14%`。移除seen竞争后仍分不开，说明问题位于unseen内部细粒度表示。
- 高频错误集中在同族类别：Blue-winged Warbler→Wilson Warbler `40`次，Baird Sparrow→Henslow Sparrow `27`次，Brandt Cormorant→Pelagic Cormorant `26`次。
- GZSL错误样本top1-top2 margin均值仅`0.604516`，正确样本为`1.725912`，错误集中在低边际近邻竞争。

决策：停止全局bias和无差别近邻调整。下一候选只能针对可由类名确定的同族类内身份方向，且不得使用真实unseen图像梯度。
