# V2-INNOVATION-012 结果

状态：`run001_marginal_rescue_pending`。

RUN-001得到`U/S/H/ZS=75.029516/79.568261/77.232263/83.028460%`，相对NCRA父模型H只提高`0.031139`个百分点。最佳位于iteration=`0`，即第一个seen训练batch之后，learned_delta=`0.049998`；后续delta增大时H普遍下降，诊断为允许幅度过大导致快速过修正。

该结果仅标记`keep_marginal`，不宣称新模块成立。模型SHA256：`8c10e7f513adc5a06300aab36e7b04d0932f8f04d46481a10f47bda1c1c4b0b2`；最后checkpoint SHA256：`3ec9dcd820608fa482c3fd6b21f48da03962d0ddbd6c172713bbb1d1c627a13f`。
