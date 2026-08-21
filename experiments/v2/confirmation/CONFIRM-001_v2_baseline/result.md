# V2-CONFIRM-001 结果

状态：`completed`。

RUN-001使用`framework/v2@3dc078c0d52bf358bf24a26e48346c97de9e99ca`、冻结配置和seed 7完成50 epoch训练。

正式结果：`U=72.655779%`、`S=75.443041%`、`H=74.023182%`、`ZS=81.534684%`，best epoch=`50`。

该结果采用`test_selected_inductive_gzsl`，明确披露official test参与选择，不是blind-test。它逐值复现历史seed 7 H，但当前仓库仍只有一个正式seed，不得扩展为多seed稳定性结论。
