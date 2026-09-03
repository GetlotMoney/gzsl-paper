# V7-ABLATION-002 结果

状态：planned。本实验是 TUNE013 CUB seen-only CE 的正式从头重训消融，不复用 off checkpoint，也不把同 checkpoint inference-off 诊断当作模块消融。

Full reference 复用已接纳 TUNE013 正式 RUN：H=79.94579718163422，code commit `35cefc52896c383e1ec75a3adc5f78d218d616a3`，config SHA `cadceb65cb9596ea8f9769424154636938120691f306766c8c769168d1f711b3`，metrics SHA `08fd38f284eced2ccd4ea49e1faae1f1163ce5b3821eb7373f9acedf9201f34e`，model SHA `7349e869c1ffef41c3cfe265486880a1c455feb442ff326ae8403bd9a9a39328`。

四个条件均从 seed7 初始化重新训练 28228 update，每 141 update official test 评估一次，各自按本条件 H 选择 best checkpoint。U/S/H 使用 200 类联合竞争，ZS 使用 50 个 unseen 类竞争；trainval seen 图像产生梯度，unseen 图像不产生梯度。
