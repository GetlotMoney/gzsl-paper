# CRA运行契约验证

V2-TRY-111使用代码`e7287ac8e24d91921eaff8152f7788ac665b3673`与seed17重跑，逐位复现`U/S/H/ZS=75.319785/84.055454/79.448210/86.219549`。

```text
metrics.json SHA256           751b87d6e04e05d043118cfd9abea357adfca9b8d69f28238a000bde403d7d2e
training.log SHA256           4826c7329b773fe3186e02a7fc171bb35ea5b434016c8e202a0243d97d0acb42
data_fingerprints.json SHA256 549886964fbef07bad2f0f65052760e57a34f8b3b26f6efca7795ba3a68d1d8e
model_best.pth SHA256         d7ac053e708037f1b43f1a8252ee9f94fa33cf302bc5bc6365ec92d77d841592
checkpoint_last.pth SHA256    d5c6cc58d04724750ea2c977502b1d8687c3b64ee3437a367b88e90481fd4576
```

输出目录：`/data/lby/projects/cv_project/GZSL_Warehouse/tries/v2/V2-TRY-111`。

## SIGTERM与resume

V2-TRY-112使用代码`8951a80e367c55bcc5375079a706f371eb5aabe4`。前两次监视尝试因训练完成过快或监视器匹配自身PID而未形成中断证据，结果目录保留但不计入结论。第三次`PARTIAL3`在epoch 5日志写出后向训练Python进程发送SIGTERM；此时原子checkpoint保持为完整epoch 4，且partial目录没有`metrics.json`。

恢复命令把该checkpoint加载到全新`V2-TRY-112`输出目录，从epoch 5继续。与同代码未中断运行比较：模型state tensor逐位相同、history完全相同、U/S/H/ZS完全相同、best epoch均为8。

```text
partial checkpoint epoch        4
partial checkpoint SHA256       b6ef9c3de1481f1cda289860f8bdd50c0836c59c6b9b992db55dfa0e11586ac6
resumed metrics.json SHA256     4022f7af8345c8d6885e12e1a36e07d0553c3076623992510f181f64bd3e24e1
resumed training.log SHA256     2479840af83a76dcba957d7ee933749825e8291a1694ddef8b927ba087a9528d
resumed model_best SHA256       4728a1a26101cf8ccbd033f8b87577dc991a8df8f864cb8a660b1d6c8d1209b0
resumed checkpoint_last SHA256  47b2d83ff1f911608f2d3673c3c8f23d795c8acd600afbca59371677ce165d93
```

本次证明同一服务器跨输出目录resume；由于当前只有`lab4090`一台训练主机，未声称跨物理机器验证。
