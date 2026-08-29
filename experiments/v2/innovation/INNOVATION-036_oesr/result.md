# V2-INNOVATION-036 结果

状态：`supported_two_seed_weak_H_candidate`。

RUN-001得到`U/S/H/ZS=76.715362/79.547596/78.105812/83.822459%`，H比OCLR高`0.033627`，但U与ZS低于OCLR，属于弱H提升而非全面改进。

最佳位于iteration=`282`，beta=`13.082210/20`，未饱和。模型SHA256：`74bd92c84278c4f623e2ae357358a34bc07a810714b1f03c236065fc77a9a8e1`；最后checkpoint SHA256：`c44ca9a8c7b9d871aaac896d6f9f41fd99f3ff77c5cf6141ef5e7b93272af4e0`。

因H差距仅0.033627，必须运行seed7判断偶然性；可靠性完成前不替代OCLR主候选。

RUN-002只把随机seed从5改为7，其余模型、输入、训练量和评估协议完全不变。

RUN-002得到`U/S/H/ZS=76.715362/79.540753/78.102514/83.822459%`，与seed5最高H只差`0.003299`。两个seed均超过OCLR最高，确认OESR的H提升不是随机批次偶然。

按owner规则主成绩取seed5最高`H=78.105812`；OESR成为当前最高H的两seed弱候选，但OCLR仍具有更高U/ZS。RUN-002模型SHA256：`2cbc9d6860c398ce938833e345fffaae6bc71b5cff6accc7dd304132c15a7bb2`；最后checkpoint SHA256：`e11280a7967c274a0dee38580c58e6798088b6fed73a7e3a3dc6e7ed03023f08`。
