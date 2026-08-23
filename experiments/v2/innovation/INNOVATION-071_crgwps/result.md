# V2-INNOVATION-071 结果

状态：`supported_two_seed_patch_free`。

RUN-001（seed5）最高`U/S/H/ZS=76.782060/80.073357/78.393178/83.886743%`，selected iteration=`423`。相对SDCR父条件H提高`0.072667`，相对S-GWPS提高`0.024811`；它是当前最高patch-free单seed H，但ZS相对父条件下降`0.067234`，需追加seed7判断可靠性。

12维selector参数均有限，中心化角色特征的跨样本标准差为`0.787403–1.293560`，没有退化为常数。

模型SHA256：`4675781801b2a67b9713a264fd3a6b30fb9dc0152351a1cf17562f342c347f2f`；最后checkpoint SHA256：`91888a00c5c33465eda550454d839d27ab66ad1d1a0d41a3d67f4a164ff5b85a`。

RUN-002（seed7）最高`U/S/H/ZS=76.705819/80.197293/78.412709/83.981764%`，selected iteration=`1833`。相对同seed SDCR提高H `0.109853`，相对同seed S-GWPS提高`0.062018`；U仅下降`0.007284`，S提高`0.237399`，ZS提高`0.061685`。

seed5/7均为正提升，最高取seed7 `H=78.412709%`。该结果比patch依赖GWPS最高`78.414246%`仅低`0.001537`，但完全不读取patch；因此晋级为两seed支持的patch-free辅助候选，尚未做相关工作检索，不声明论文核心原创。

RUN-002模型SHA256：`8ea92cce1560e0fcda58bd8f3bb245a0cc6b576ac0cd78e26fbca5209f9f2b9a`；最后checkpoint SHA256：`d4a1f06c4f30ebdc481ba8bc929dd014e7ccc318577c69167e16e0b391ca3a9d`。
