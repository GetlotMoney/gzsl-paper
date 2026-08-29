# V2-INNOVATION-007 结果

```text
highest H / seed        = 80.474080 / 17
H mean/min/max/range    = 80.228888 / 79.917063 / 80.474080 / 0.557017
U mean                  = 76.999331
S mean                  = 83.743058
ZS mean                 = 86.832157
Delta H min/max         = 0.392566 / 0.930470
```

seed17链式消融：CRA=`79.448210`，CRA+VPA=`79.543609`，CRA+EBC=`79.791176`，CRA+VPA+EBC=`80.474080`。VPA主要提高ZS，EBC主要恢复U/S平衡，二者互补。正反ridge与校准均有传统方法先例，因此这里只声明当前框架中的可靠组合，不作核心原创claim。

4seed固定等权ensemble的H为`80.115795%`，没有超过最高单模型或四seed均值，因此最终推理不使用ensemble。

VPA reverse ridge=0.1时VEBC H为`80.165438%`，低于0.01条件`80.474080%`；最终固定0.01。
