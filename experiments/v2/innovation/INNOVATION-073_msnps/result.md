# V2-INNOVATION-073 结果

状态：`supported_two_seed_patch_free`。

RUN-001（seed5）最高`U/S/H/ZS=76.810980/80.179805/78.459247/83.981764%`，selected iteration=`846`。相对C-RGWPS提高H `0.066069`，但比union-SNPS最高低`0.021463`。

mutual top-5把语义边从798收紧到202，有效pair从4691降至4299；四项均超过SDCR父条件。追加seed7检验是否修复union-SNPS的增量方向不一致。

模型SHA256：`fd8355eea9584e782fa41715eb85ee7bde4a53058fc55512edcbbe5354ba18f0`；最后checkpoint SHA256：`423e871b14b5291b4d8444a8bf13f009aa3044580105d5673b0608b72ca1b0bb`。

RUN-002（seed7）最高`U/S/H/ZS=76.849276/80.070651/78.426898/83.987314%`，selected iteration=`282`；相对同seed C-RGWPS提高H `0.014189`，相对SDCR提高`0.124042`。

M-SNPS相对C-RGWPS的seed5/7增量为`+0.066069/+0.014189`，两seed同号；自身H范围仅`0.032349`。mutual规则成功修复union-SNPS的增量方向不一致，晋级为稳定patch-free辅助候选。最高H仍取SNPS seed5 `78.480710%`。

RUN-002模型SHA256：`75e7764d8adc95100cce2df4330e0746136a0d6b897b9eebcc49a6f2f6d36f7f`；最后checkpoint SHA256：`d089719dbeccbd4f8813c56b5578b6d81c3b98799ab8cb958ad982d9b3efdd18`。
