# V4-CONFIRM-003 PCLR跨数据集单轮双Agent审核 receipt

- 冻结代码：`d1d8940844e2e540234a7cb8fae688d80efb5247`；tree：
  `10655d2c40f4af665592c9acceb57353322fa83f`。
- AWA2 config/asset SHA：`4c636b...c88` / `f93c9a...adb2`；动态轴
  class/seen/edge=`50/40/117`。
- SUN config/asset SHA：`43e5f5...356` / `385744...93b8`；动态轴
  class/seen/edge=`717/645/1633`。
- 共享本地证据：相关`29 passed`、`py_compile`、`git diff --check`通过。

A/B独立审查后直接交叉发现初始P1：关系资产只绑定SHA但未证明动态图语义合法。集中修复后，
loader强制manifest动态class/seen/E/direction/embed/graph source/required outputs，保留并校验
FP32/int64、shape/finite/unit norm、edge范围、a<b、无自环、unique和全类degree覆盖；测试包含
class mismatch、negative endpoint、self-loop和duplicate。原两名Reviewer复核后P0/P1归零。

GPU0 AWA2完整micro：Raw H=`96.372177`精确复现，Full
`U/S/H/ZS=97.656298/95.577794/96.605867/99.047691`，`ΔH=+0.233690`、
`ΔZS=-0.125551`、gap `2.3338→2.0785`，gate=true，metrics SHA=`7df405...7000`。

GPU1 SUN完整micro：Raw H=`73.341746`精确复现，Full
`U/S/H/ZS=80.763888/68.837208/74.325131/92.847222`，`ΔH=+0.983386`、
`ΔZS=+0.347227`、gap `12.6502→11.9267`，gate=true，metrics SHA=`09a232...66b9`。

两名Reviewer直接互认双设备micro，最终`P0=0/P1=0`，共同结论：

**代码单轮双Agent对抗审核通过**

两数据集均为generic class-name directions，`llm_world_knowledge_used=false`；不能与CUB的LLM
形态差异句混称同一文本资产。结果明确nested official-test selection、nonblind、无unseen梯度。
