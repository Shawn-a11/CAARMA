# CAARMA 实验清单

> 记录每个实验：分支、配置、状态、结果、CSV 路径。
> CSV 缺失的实验需要从服务器粘贴 30-epoch metrics。

## 命名约定

```
exp/innovation-{innovation_name}-{single|ddp}
ckpt save_dir: /root/autodl-tmp/CAARMA/caarma_mfa_ckpts_{tag}_{single|ddp}
csv:           output/csv/{exp_id}_{tag}.csv
plot:          output/plots/{exp_id}_{tag}.{png,pdf}
```

---

## 实验汇总（完成情况一览）

| ID | Tag | 分支 | 硬件 | Best EER | Best Ep | CSV | vs paper 3.09% |
|----|-----|------|------|---------|--------|-----|---------------|
| 01 | singlegpu_baseline | `exp/rewrite-algorithm2-correct-gan` | 单卡 V100 | **3.51%** | 19 | ✅ | +0.42% |
| 03 | ddp_baseline | `exp/ddp-4gpu-v100` | 4 卡 DDP | **3.48%** | 17 | ✅ | +0.39% |
| 04 | ddp_source_code | `exp/ddp-4gpu-v100-source_code` | 4 卡 DDP | 3.83% | 26 | ✅ | +0.74% ❌ |
| 05 | ddp_slerp | `exp/innovation-slerp-ddp` | 4 卡 DDP | 3.58% | 21 | ✅ | +0.49% ❌ |
| 06 | ddp_wavlm | `exp/innovation-wavlm-ddp` | 4 卡 DDP | 3.71% | 27 | ✅ | +0.62% ❌ |
| **07** | **ddp_specaug** | `exp/innovation-specaug-ddp` | 4 卡 DDP | **3.27%** | 21 | ✅ | **+0.18%** ⭐ |
| 08 | ddp_full_aug (killed) | `exp/innovation-full-aug-ddp` | 4 卡 DDP | 4.36% | 24 | ✅ (ep1-26) | +1.27% ❌（已弃） |
| 09 | ddp_aug_per_sample | `exp/innovation-aug-per-sample-ddp` | 4 卡 DDP | **3.34%** | 25 | ✅ | +0.25% ⚠️ |
| 10 | ddp_redimnet_b6 | `exp/innovation-redimnet-b6-ddp` | 4 卡 DDP | — | — | — | ⏳ 待启动 |
| 11 | ddp_vmf | `exp/innovation-vmf-ddp` | 4 卡 DDP | — | — | — | ⏳ 待启动 |
| **12** | **plain_baseline_no_gan** | `exp/baseline-plain-mfa-ddp` | 4 卡 DDP | **3.61%** | 15 | ✅ | vs paper baseline 3.33% → **+0.28%** |

---

## Key Findings（按重要性排序）

### 0. 纯 MFA baseline（无 GAN 无增强）= 3.61%，揭示「配方偏移」与「GAN 真实增益」

复现论文 Table 2 ID1 的纯 baseline（编码器 + AM-Softmax，无判别器/无 mixup/无合成损失/无增强），
得到 **3.61% @ ep15**。这是之前实验矩阵里唯一空着的格子，一跑就把两件事拆开了：

| | EER | 说明 |
|---|-----|------|
| Paper 纯 baseline (Table 2 ID1) | 3.33 | 论文锚点 |
| **我们的纯 baseline (12)** | **3.61** | 比论文高 **+0.28** = 恒定「配方偏移」 |
| + 冻结 HuBERT GAN (03) | 3.48 | 冻结版 GAN 只买了 **−0.13** |
| + 冻结 GAN + SpecAug (07) | 3.27 | 再 −0.21 |
| Paper 全家桶 (Table 2 ID6) | 3.09 | 论文 SOTA |

→ **配方偏移 +0.28 与 GAN 无关**：增强已被排除（MFA-Conformer 原论文明写
"No ... augmentation"，CAARMA 照搬其配方），最大嫌疑是有效 batch（我们 4×50=200，
论文写 50，若作者实际 effective 50 则差 4 倍）+ seed 方差。
→ **冻结 HuBERT 的判别器只交付了 0.13（论文 GAN 机制值 0.24 的一半）**，
量化印证「冻结 backbone = 残废判别器」，为主实验 `exp/repro-3.09-one-shot`（解冻 HuBERT）
提供了明确对照基线。
→ **过拟合证据**：纯 baseline 在 ep15 触底后回升（train acc 100%、am_loss→0.13，
EER 3.61→3.72），说明 CAARMA 的合成类在小规模 Vox1 上**部分起正则化作用**——这正是
论文「class scarcity」论点的直接可视化（图 `12_plain_baseline_vs_caarma_ladder`）。

### 1. SpecAugment 是 0.21% baseline gap 的主要解释

| | EER |
|---|-----|
| ddp_baseline（无 augmentation） | 3.48% |
| **ddp_specaug（仅 SpecAug）** | **3.27%** |
| Paper SOTA | 3.09% |

→ SpecAug 单独把我们的 baseline 从 3.48% 推到 3.27%，**关闭 54% 的 paper gap**。CAARMA 论文未提 augmentation，但加上后**接近 paper 数字**。

### 2. 所有架构 / 训练流程创新都未超越 baseline

| 实验 | 改动 | 结果 |
|------|------|------|
| 04 source_code | pretrain + 5:1 G:D + d_loss/2 | 3.83% ❌（-0.35%） |
| 05 slerp | 球面插值替代 LERP | 3.58% ❌（-0.10%） |
| 06 wavlm | WavLM 替换 HuBERT D | 3.71% ❌（-0.23%） |

→ 在我们 setup 下，paper Algorithm 2 加上原生 CAARMA pipeline 已是局部最优。这三个改动都减分。**augmentation 才是真正的杠杆**。

### 3. Augmentation 必须用正确的「per-sample 单选」策略

| 实验 | 策略 | Best EER |
|------|------|---------|
| 07 specaug | SpecAug only | 3.27% ✅ |
| 08 full_aug (killed) | 100% noise + 100% reverb + SpecAug 全部叠加 | 4.36% ❌ |
| 09 aug_per_sample | per-sample 随机选 {clean / noise / reverb} + SpecAug | **3.34%** ⚠️ |

→ 双叠加 augmentation 不是「更多更好」，而是**直接破坏训练**。标准 SV 配方是 per-sample 单选（ECAPA-TDNN / MFA-Conformer 引用）。
→ **意外发现**：即便用了正确的 per-sample 单选策略，MUSAN+RIR 加在 SpecAug 之上仍然略差（3.34% vs SpecAug-only 3.27%）。说明在 CAARMA pipeline + MFA-Conformer 下，**waveform-level augmentation 与 SpecAug 没有协同效应**，SpecAug 已经够用。这与 ECAPA-TDNN 经验（waveform aug 有显著贡献）不同，可能与 CAARMA 的 embedding-level mixup 已经提供了足够的扰动有关。

---

## 缺失的关键实验

### ~~09 — ddp_aug_per_sample~~ ✅ 已完成

- 结果：3.34% @ ep 25 — 比 SpecAug-only (3.27%) 略差
- 结论：waveform-level aug 在 CAARMA pipeline 下不补救 SpecAug 的剩余 gap

### 10 — ddp_redimnet_b6

- 分支：`exp/innovation-redimnet-b6-ddp`
- 改动：MFA-Conformer (19.8M) → ReDimNet-b6 (15M, IDRnD 2024)
- 验证假设：更强 encoder 在 CAARMA pipeline 下是否提升
- 预期 EER：未知
- 预计耗时：8h

### 11 — ddp_vmf

- 分支：`exp/innovation-vmf-ddp`
- 改动：mixup 中点 → vMF(midpoint, κ=50) 球面采样
- 验证假设：vMF 多样性是否优于 SLERP 单点
- 预期 EER：未知
- 预计耗时：8h

---

## 当前最强 baseline 用于后续创新对照

**新创新点应该 vs `exp/innovation-specaug-ddp` (3.27%) 对照**（最强可复现 baseline，包含 modern SV 默认开启的 SpecAug）。

之前的 SLERP / WavLM 实验是在 `exp/ddp-4gpu-v100` (3.48%) 上跑的，结果都比 baseline 差，重跑在 SpecAug baseline 上的预期收益不大（除非 SLERP 与 SpecAug 有协同）。

---

## 输出文件结构

```
output/
├── EXPERIMENTS.md                       ← 本文件
├── csv/
│   ├── 01_singlegpu_baseline.csv        ✅
│   ├── 03_ddp_baseline.csv              ✅
│   ├── 04_ddp_source_code.csv           ✅
│   ├── 05_ddp_slerp.csv                 ✅
│   ├── 06_ddp_wavlm.csv                 ✅
│   ├── 07_ddp_specaug.csv               ✅
│   └── 08_ddp_full_aug.csv              ✅ (only ep1-26, killed)
└── plots/
    ├── 01_singlegpu_baseline_paper_faithful_ep5-30.{png,pdf}
    ├── 02_singlegpu_right_algorithm2_ep1-30.{png,pdf}
    ├── 03_ddp_baseline_right_algorithm2.{png,pdf}
    ├── 04_ddp_source_code_pretrain_5to1.{png,pdf}
    ├── 05_ddp_slerp_innovation.{png,pdf}
    └── 06_ddp_3way_compare_baseline_source_slerp.{png,pdf}
```

需补充的图：
- 07_ddp_specaug.{png,pdf}（单图）
- 08_ddp_full_aug.{png,pdf}（单图，反面教材）
- 09_unified_7way_compare.{png,pdf}（所有 7 个完成实验同图叠合）
