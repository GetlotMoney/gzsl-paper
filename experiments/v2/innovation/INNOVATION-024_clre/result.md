# V2-INNOVATION-024 结果

状态：`supported_candidate_provenance_pending`。

RUN-001得到`U/S/H/ZS=75.997263/79.707325/77.808093/83.523118%`，相对SEBC父模型变化为`+0.224704/+0.360775/+0.289711/+0.461334`个百分点，四项同时提高；H也比此前CCPE最高`77.666533`高`0.141560`。

最佳位于iteration=`1974`，beta=`19.266289/20`，未越过98%饱和门槛。训练只用seen图像CE，unseen文本允许使用、unseen图像不进入梯度；official test按陈式规则选一个全局best。

模型URI：`/data/lby/projects/cv_project/GZSL_Warehouse/innovation/v2/INNOVATION-024_clre/RUN-001/model_best.pth`

模型SHA256：`03db81c9e42080eba45f788087ad7c3845ee0c0128135b2fbc7a91a1d2cf8538`；最后checkpoint SHA256：`83c7620e55eced8d8e1ea3591fec8db3a1e94c23d6e3e86732e604f190f07d73`。

CLRE作为supported创新候选保留；Claude cache的准确prompt/编码模型来源尚不完整，且最近相关工作未检索，因此暂不作原创核心claim。
