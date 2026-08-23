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

RUN-004使用25分位门槛`0.236863`，seen/unseen gate=`0.074414/0.087582`。最高`U/S/H/ZS=76.681465/80.107862/78.357224/83.888441%`，比中位数AGCT提高H `0.017701`，比SDCR提高`0.036714`；beta=`-2.309143`未饱和。追加seed7可靠性后再替换正式条件。

RUN-004模型SHA256：`ed0c8d3eb655d669a410889205f485ce91747734bfd9d551a8cfbddd308cf0fc`；最后checkpoint SHA256：`b2833ee7d52d7328f77b8b70ebac7ebfb22846384a3c1454da607482aabaf1fe`。

RUN-005使用seed7父链，25分位门槛=`0.237726`、unseen gate=`0.087829`。最高`U/S/H/ZS=76.647568/80.107862/78.339523/83.854544%`，相对seed7 SDCR提高H `0.036667`；beta=`-2.256506`未饱和。

25分位两seed均正，正式替换中位数条件；主成绩按owner规则取seed5 `H=78.357224%`。门槛宽度轴关闭，最后一次方法级补救只把gate温度从0.1降到0.05。

RUN-005模型SHA256：`cb1f2d4dff1e3b0653308e13d29d6a97a6a70b64b3175c78de27e97c863131a6`；最后checkpoint SHA256：`c644449a656a5b67db1d1a9d07c58f4cfeff536593ab9c357698d13830fdd422`。

RUN-006固定25分位并把gate温度从0.1降到0.05，最高指标与RUN-004逐项相同：`76.681465/80.107862/78.357224/83.888441%`。更硬门控没有额外收益，最终保留已做两seed验证的温度0.1条件并关闭AGCT参数轴。

RUN-006模型SHA256：`ab9b0b01774d3633f67026b9929b21317c88eea89a87a0ebc79597d529b0e234`；最后checkpoint SHA256：`67407c3145d259b979d8ad96f759fa4b249396bd14eaed65e70fe5ed72ee317a`。
