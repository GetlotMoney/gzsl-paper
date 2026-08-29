# V2-INNOVATION-025 结果

状态：`rejected_not_complementary`。

两分支直接相加的初始结果为`U/S/H/ZS=76.074433/79.125082/77.569776/83.362424%`，低于CLRE `77.808093`。训练将patch scale降至`0.860083`后，最高为`76.067758/79.295385/77.648045/83.322996%`，仍未超过CLRE。

Claude全局证据与CCPE局部证据在当前权重下不互补，IDEA-059拒绝。模型SHA256：`0ea35f1f4edc0ef94d4501ccc4bb394681549bc256338894b1c9c419bdcdcc49`；最后checkpoint SHA256：`a20134d78083dd778a27e9cbf587d05e59a9a54a6b5ee6b3338f26cc371ffce4`。
