# V2-INNOVATION-045 结果

状态：`retained_saturated_signal_rescue_1_failed`。

RUN-001得到`U/S/H/ZS=76.748133/80.051959/78.365239/83.953977%`，best位于iteration `2397`，相对SDCR父模型H提高`0.044729`个百分点。class variation=`0.007255`、最小权重=`0.039332`，说明共享规则产生了非塌缩的类别差异。

但logit residual范围达到`[-0.249994,0.249999]`，触及预设±0.25边界；当前只保留为正信号，不晋级。RESCUE-1只把残差上限收紧到±0.10，其他结构、数据、seed和训练量不变。

RUN-001模型SHA256：`6a07fde7e1d76bde4aa02340315a782a741127b0240ff525b558f406a7633b8c`；最后checkpoint SHA256：`d2bbcd117c0a179cffd854e8ce9fcdf4c7c34e257fb582ef88b667a516600bea`。真实unseen图像未进入梯度，official test评估202次并用于选模。

RUN-002把上限收紧到±0.10，得到`U/S/H/ZS=76.780897/79.959893/78.338157/83.953977%`，低于RUN-001，且残差仍达到`[-0.099999,0.099998]`。因此“继续收紧幅度”不是正确补救，参数轴关闭。模型SHA256：`9d6bd9d974cb11545c15f489ab38dc1b5384da90a114489624ad3a67d5177612`；最后checkpoint SHA256：`47840735a3947af472dc9cac565cd4cf866b3dd55ecfb1b2e183015fae1df14e`。
