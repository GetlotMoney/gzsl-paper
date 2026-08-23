# V2-INNOVATION-058 结果

状态：`supported_two_seed_weak_gain`。

RUN-001使用seed5父链，train-only门槛由`416`个“错误且top2同族”样本的margin中位数固定为`0.542624`。最高`U/S/H/ZS=76.647568/80.107862/78.339523/83.888441%`，相对SDCR H提高`0.019013`个百分点。

best beta=`-2.033723`未饱和；seen/unseen GZSL平均gate=`0.152989/0.182175`，门控真实生效。增益很小且U/ZS略降，暂只保留弱正信号，追加seed7可靠性。

模型SHA256：`ae6c61ac0186184a6e2477065d99ad9c7f19cb49f805b31ebba57e007f27eaed`；最后checkpoint SHA256：`1e4e80528962135ed7fba0f10001f55d4d26e7c3f9622c990c5797304041093c`。

RUN-002使用seed7父链，train错误同族样本`414`个，门槛=`0.547560`。最高`U/S/H/ZS=76.647568/80.107862/78.339523/83.854544%`，相对seed7 SDCR H提高`0.036667`个百分点；beta=`-1.962573`、unseen gate=`0.183605`。

两seed均为正增益且最高H完全一致，AGCT标记supported辅助候选；主成绩按owner规则取`H=78.339523%`。增益很小且U/ZS略降，不作为论文核心创新，下一Experiment检验负beta的“反共识”触发条件能否放大收益。

RUN-002模型SHA256：`7d4f4d10243ed1845c6081be82882b31ba78911927adebfb5994d983c60cdd51`；最后checkpoint SHA256：`affda041196eba37e5df03da116435f78d1f95e53058353915a44c7f566ad08c`。

RUN-003把train错误margin门槛从中位数提高到75分位，阈值=`0.984965`，seen/unseen gate扩大到`0.239759/0.305713`。所有非零条件均低于父模型，best退回`H=78.320510%`、beta=0，说明覆盖过宽引入噪声；原中位数AGCT继续保持正式supported条件。

RUN-003模型SHA256：`6668426bb89dbc54e0a4a460552ffde62d93d47ecf008baf54398e68f1c2933d`；最后checkpoint SHA256：`a9137e5307925370e7574e60c4f1c7693631d7d3819b7fdc2bbc4ab1efeb9437`。门槛轴最后检查25分位窄覆盖。
