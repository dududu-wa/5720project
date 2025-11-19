下面是一套**可直接落地**的 CIFAR-10 项目完整方案：从结构、数据来源、算法选择、训练细节、复现与汇报都给到。所有外链与“最佳实践”均附权威/官方来源。

# 1) 项目目标与评分导向

* 目标：在 **CIFAR-10**（32×32，10类）上训练图像分类器，尽可能提升 Top-1 准确率；要求代码可复现、汇报清晰。官方数据概述：60,000 张图像，其中 50k 训练、10k 测试。([cs.toronto.edu][1])
* 课程/平台常用数据接口：**torchvision.datasets.CIFAR10**（一行下载+缓存，含 transforms 管道）。([PyTorch Docs][2])

---

# 2) 数据与链接（两种获取方式）

* 官方主页（.tar.gz）：**CIFAR-10 and CIFAR-100 datasets**（U Toronto）。适合离线下载/校验。([cs.toronto.edu][1])
* 代码自动下载：**torchvision.datasets.CIFAR10(root, download=True)**。适合脚本端到端拉取。([PyTorch Docs][2])
  （备用：UCI、Kaggle 镜像与竞赛页，非必须。([archive.ics.uci.edu][3])）

---

# 3) 推荐模型与增强组合（高分“天花板级”配方）

* **主干网络（择一）**

  * **WideResNet-28-10 / 40-4**：CIFAR 上的强 CNN 基线，官方/原作者实现与多套 PyTorch 复现可用。([GitHub][4])
  * 可选替代：PyramidNet / Shake-Shake（更激进，但复现复杂）。
* **数据增强 / 正则**

  * **RandAugment**（只调 N、M 两参，无需搜索，工程落地友好）。([GitHub][5])
  * **Mixup**（FB official repo & 论文实现）。([GitHub][6])
  * **CutMix**（数据集包装/损失封装现成，含 CIFAR-10 结果与训练细节）。([GitHub][7])
  * 可选：AutoAugment（论文给出 CIFAR-10 SOTA 级别结果，代价是策略搜索，实践一般用 RandAugment 替代）。([CVF开放获取][8])
  * 可选：Fast AutoAugment（更高效的搜索/复现）。([NeurIPS Papers][9])

> 经验法则：**WRN + RandAugment + Mixup/CutMix + Cosine LR**，在公开复现实验通常 ≥98% Top-1，调参到位可更高；课程环境无需外部预训练即可达成。支持证据见各方法仓库/论文的 CIFAR-10 结果区。([GitHub][10])

---

# 4) 项目目录结构（建议）

```
cifar10-project/
├── configs/
│   └── wrn28x10_ra_mixup.yaml
├── data/                      # torchvision 自动下载到此（或 ~/.torch）
├── src/
│   ├── models/wrn.py          # 或直接引用第三方实现
│   ├── augment/randaugment.py # 可 vendor 第三方实现
│   ├── augment/mixup_cutmix.py
│   ├── dataset.py             # CIFAR10 + transforms
│   ├── train.py               # 训练循环/EMA/混合精度
│   ├── eval.py                # 测试集评估/混淆矩阵/TTA(可选)
│   └── utils.py               # 日志、保存、随机种子、指标
├── run.sh                     # 一键训练命令
├── requirements.txt
├── README.md                  # 复现命令、环境、结果表
└── report/
    └── slides.pptx / report.pdf
```

---

# 5) 环境与依赖

* **Python 3.10+，PyTorch + torchvision**（与 CUDA 匹配的版本），tqdm、tensorboard/Weights & Biases（二选一）等。
* 数据自动下载：使用 **torchvision.datasets.CIFAR10**。([PyTorch Docs][2])

`requirements.txt` 示例：

```
torch==<match your cuda>
torchvision==<same>
tqdm
tensorboard
einops
```

---

# 6) 数据管道（归一化与增强）

* 官方统计量（常用）：**mean=[0.4914, 0.4822, 0.4465]，std=[0.2023, 0.1994, 0.2010]**。
* Train transforms：RandomCrop(32, padding=4) + RandomHorizontalFlip + RandAugment(N=2, M=9) + ToTensor + Normalize。([PyTorch Docs][2])
* Test transforms：ToTensor + Normalize。
* Mixup/CutMix 在 **collate_fn** 或 **训练 step** 中应用（保持 val/test 纯净）。([GitHub][6])

---

# 7) 训练设置（建议默认值）

* **Backbone**：WRN-28-10（通道放大 10，深度 28）。([GitHub][11])
* **优化器**：SGD(momentum=0.9, weight_decay=5e-4)。
* **学习率**：base LR=0.1（bs=128），**CosineAnnealingLR**，warmup 5 epochs。
* **批大小**：128（单卡 8–12GB 可跑），多卡线性放大并等比例调 LR。
* **历元**：200（CutMix/Mixup 训练 200–300 epoch 常见；可视 GPU 时间调节）。([GitHub][10])
* **正则**：Label Smoothing=0.1（与 Mixup/CutMix 二选一或小幅并用）。
* **EMA**（Exponential Moving Average）权重：decay=0.999（可带来 0.1–0.3% 稳定收益）。
* **Mixed Precision**：AMP fp16（减少显存/提速）。
* **保存策略**：保存 **best-val-acc** 与 **last**；日志记录 LR/损失/准确率。

---

# 8) 可运行命令（样例）

`run.sh`：

```bash
#!/usr/bin/env bash
EXP=wrn28x10_ra_mixup
python -m src.train \
  --dataset cifar10 --data_root ./data \
  --model wrn28x10 \
  --epochs 200 --batch_size 128 \
  --opt sgd --lr 0.1 --momentum 0.9 --wd 5e-4 \
  --sched cosine --warmup 5 \
  --randaugment N=2 M=9 \
  --mixup 0.2 \
  --label_smoothing 0.1 \
  --ema 0.999 \
  --amp \
  --seed 42 \
  --out runs/${EXP}
python -m src.eval --ckpt runs/${EXP}/best.ckpt --data_root ./data --dataset cifar10
```

---

# 9) 评估与可视化

* **指标**：Top-1 Acc（测试集），训练/验证曲线；**混淆矩阵**；误分类样例。
* **日志**：tensorboard 或 W&B；固定随机种子与版本，记录显卡/时长。
* **TTA**（可选）：水平翻转 + 多裁剪，对测试集少量增益但要声明。

---

# 10) 复现性与消融（建议在 README 与报告中呈现）

* **随机种子**、**环境版本**、**下载镜像**、**精确命令**、**准确率表**。
* **消融**：Baseline（无增强）→ +RandAug → +Mixup/CutMix → +EMA/Label Smoothing → （可选）更深 WRN。
* 记录每一步的测试准确率与提升，附训练曲线。

---

# 11) 里程碑与分工（两周样例）

* **D1–D2**：跑通 baseline（WRN-28-10 + 标准增强），≥90%。
* **D3–D6**：接入 RandAugment、Mixup/CutMix、Cosine/Warmup，找到稳定超参。([GitHub][5])
* **D7–D10**：长程训练（200–300 epoch），做消融与曲线、混淆矩阵。([GitHub][10])
* **D11–D12**：固化脚本与 README，撰写 5 分钟汇报（方法/结果/分工）。
* **D13–D14**：最终复现一次，导出 best ckpt 与日志，准备答辩。

---

# 12) 参考实现与仓库（直接可用/可 vendor）

* **WideResNet（作者/复现）**：([GitHub][4])
* **RandAugment（PyTorch 实现）**：([GitHub][5])
* **Mixup（官方/复现）**：([GitHub][6])
* **CutMix（数据集封装/损失）**：([GitHub][7])
* **CutMix 结果与训练细节（CIFAR-10）**：([GitHub][10])
* **AutoAugment（论文 PDF）**：([CVF开放获取][8])
* **Fast AutoAugment（论文 PDF）**：([NeurIPS Papers][9])
* **CIFAR-10 官方页 / torchvision 文档**：([cs.toronto.edu][1])

---

## 你可以现在就做的两件事

1. gpu为3060latop 把上述配方打成**可运行脚手架**（含 `train.py / eval.py / augment/ / models/ / run.sh / README`），默认跑 **WRN-28-10 + RandAugment(N=2,M=9) + Mixup(α=0.2) + Cosine**。
2. 如果你更想“快交”，我也可以削减到 **ResNet-18 + 轻增强** 的**极速版**（<1 天完成基线+对比图），然后再逐步叠加增强拿分。

[1]: https://www.cs.toronto.edu/~kriz/cifar.html?utm_source=chatgpt.com "CIFAR-10 and CIFAR-100 datasets"
[2]: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html?utm_source=chatgpt.com "CIFAR10 — Torchvision main documentation"
[3]: https://archive.ics.uci.edu/dataset/691/cifar%2B10?utm_source=chatgpt.com "CIFAR-10"
[4]: https://github.com/szagoruyko/wide-residual-networks?utm_source=chatgpt.com "szagoruyko/wide-residual-networks: 3.8% and 18.3% on ..."
[5]: https://github.com/ildoonet/pytorch-randaugment?utm_source=chatgpt.com "Unofficial PyTorch Reimplementation of RandAugment."
[6]: https://github.com/facebookresearch/mixup-cifar10?utm_source=chatgpt.com "Mixup-CIFAR10 - Beyond Empirical Risk Minimization"
[7]: https://github.com/ildoonet/cutmix?utm_source=chatgpt.com "ildoonet/cutmix: a Ready-to-use PyTorch Extension ..."
[8]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf?utm_source=chatgpt.com "AutoAugment: Learning Augmentation Strategies From Data"
[9]: https://papers.neurips.cc/paper/8892-fast-autoaugment.pdf?utm_source=chatgpt.com "Fast AutoAugment"
[10]: https://github.com/hysts/pytorch_cutmix?utm_source=chatgpt.com "hysts/pytorch_cutmix: A PyTorch implementation of CutMix"
[11]: https://github.com/bmsookim/wide-resnet.pytorch?utm_source=chatgpt.com "bmsookim/wide-resnet.pytorch: Best CIFAR-10, CIFAR-100 ..."
