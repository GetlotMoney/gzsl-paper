# IDEA-130：JEDS Jackknife证据选择器

status: testing
problem: S-EDPS每batch只循环屏蔽一个证据，不同证据与随机batch内容偶然绑定，可能造成剩余方差。
hypothesis: 每个batch平均全部11种单证据缺失CE，可消除mask-batch耦合，同时保留不同视图的有用差异。
evidence_refs: IDEA-128的S-EDPS低学习率两seed有效；IDEA-129证明强制视图一致有害，因此改为只平均监督loss。
base_commit: 630c7bbb4edf1bf2eff1eea4e97ad9c038865f2a
core_change: 每batch从单一循环mask改为11种leave-one-evidence-out视图的平均pair CE。
success_condition: seed5 H超过S-EDPS 78.572828；通过后追加seed7。
failure_condition: H不超过S-EDPS或计算代价显著而无增益；最多三次方法级补救。
experiment: V2-INNOVATION-097
paper_core_innovation: false
