# V2-TUNE-003 结果

状态：`topology_tuning`。

每折严格执行：100类中的80%图像参与梯度；同100类的20%图像作为val-seen；剩余50类全部图像作为val-unseen。pseudo-unseen图像不进入loss或backward，official test不加载。

RUN-001建立三模块当前超参数基准：`topology_weight=0.1 / max_transport_step=1.5 / max_generator_magnitude=0.2`。选择指标是三个fold同一epoch的mean H，并同时记录min/max/range、mean U和mean ZS。

RUN-001在统一epoch=`17`达到`mean U/S/H/ZS=74.925892/77.350948/76.113854/79.673799%`，fold H=`75.902644/75.703050/76.735867%`，min/max/range=`75.703050/76.735867/1.032817`。该条件固定为topology调参基线。

模型SHA256：`7ce2cc0a4dc4d16ff8cee9ca3bd380c0e41bff92cb848c5e43ea42e5376128a4`；最后checkpoint SHA256相同。official test未加载。

RUN-002/003预注册为topology轴：仅把`topology_weight`改为`0.03/0.2`，其余配置和fold逐项相同。

RUN-002（topology=`0.03`）在epoch 17得到`mean U/S/H/ZS=75.125722/77.030581/76.061602/79.762876%`，H低于基线`0.052252`且range扩大到`1.746188`，淘汰。

RUN-003（topology=`0.2`）在epoch 24得到`mean U/S/H/ZS=74.431213/78.602773/76.460115/79.591872%`，H提高`0.346262`且range降为`0.753863`，但U/ZS分别低于基线`0.494679/0.081927`。按泛化约束不保留，topology轴固定`0.1`。

RUN-002模型SHA256：`9d6566ccbc28e243603f1c89d744e3bf6e800ac356c1d5824f5c383a1028178f`；RUN-003模型SHA256：`23f9f945307dfe2d340bc7b5cb1b724aec93ed7fff2df3936ba41c56c366a499`；各自checkpoint SHA256相同。

RUN-004/005预注册transport轴：固定topology=`0.1`和CCGR幅度=`0.2`，只把`max_transport_step`改为`0.5/1.0`。

RUN-004（transport=`0.5`）在epoch 17得到`mean U/S/H/ZS=78.943970/70.611630/74.539610/81.470446%`，明显偏向U并破坏S/H，淘汰。

RUN-005（transport=`1.0`）在epoch 17得到`mean U/S/H/ZS=75.902977/75.687253/75.792818/79.824275%`，U/ZS高于基线但H低`0.321036`，不保留。transport轴固定`1.5`。

RUN-004模型SHA256：`8c6ce80566b0052825361760ae8ba913e51af7876745f9d7fdc3cb60362eb148`；RUN-005模型SHA256：`b975bf43dedc7e36dc20101cfdcda1f5e47aa97925e050ad6bdb9c95f30d6533`；各自checkpoint SHA256相同。
