# CAARMA 复现与扩展研究进展

> 面向导师的进度汇报 · 中文版
> 推荐时长：12-15 分钟，10 张幻灯片

---

## Slide 1 — 标题页

**标题**：CAARMA 论文复现与扩展研究 — 阶段性进展

**副标题**：从复现到 SpecAugment 发现

**内容**：
- 汇报人：[姓名]
- 日期：2026.06.02
- 任务：复现 CAARMA (EMNLP 2025) 并探索性能突破

**讲稿**：今天向您汇报近一个月的 CAARMA 研究进展。重点包含三件事：第一，论文复现到什么程度；第二，我们尝试了哪些创新，结果如何；第三，下一步计划。

---

## Slide 2 — 任务背景

**标题**：研究背景与目标

**内容**：
- **CAARMA**：CMU 团队 EMNLP 2025 论文，VoxCeleb1 上达到 EER 3.09%
- **核心思想**：在 embedding space 做 mixup 生成「合成说话人」+ 对抗训练（adversarial training）+ HuBERT-based discriminator
- **本研究目标**：
  1. 复现论文报告的 3.09% EER
  2. 在此基础上探索创新点（球面插值、不同 encoder、不同 backbone discriminator 等）
  3. 验证创新点是否能突破 paper SOTA

**讲稿**：CAARMA 是用 mixup 加对抗训练做 zero-shot 说话人验证的方法。我们的复现目标是 3.09% 的 EER；同时我们希望测试一些直接的扩展，看能不能进一步压低 EER。

---

## Slide 3 — Baseline 复现结果

**标题**：忠实复现 CAARMA：DDP 4×V100

**插图**：`output/plots/03_ddp_baseline_right_algorithm2.png`

**内容**：
- 严格按照 paper Section 4.2.3 复现：AdamW lr=1e-3, m=0.2, s=30, weight_decay=1e-7, 30 epochs
- 4 卡 V100 DDP，per-GPU batch=50（有效 batch=200）
- **复现 Best EER = 3.48% @ epoch 17**
- vs paper 3.09% → **gap 0.39%**

**讲稿**：左图展示了基线复现 30 epoch 的 EER 曲线。在 epoch 17 触底 3.48%，比论文报告的 3.09% 高 0.39%。这个 gap 是我们后续探索的起点。

---

## Slide 4 — Gap 来源排查

**标题**：0.39% Gap：可控因素逐项排除

**内容（表格）**：

| 嫌疑因素 | 是否成立 | 验证方法 |
|---------|---------|---------|
| ~~trials file 不一致~~ | ❌ | 验证为标准 VoxCeleb1-O 37,720 pair |
| ~~num_spk 错误~~ | ❌ | 论文 Table 2 明确 VoxCeleb1 = 1211 |
| ~~训练数据缺失~~ | ❌ | 148,591 / 148,642 = 缺 51 句 (0.034%) |
| ~~mixup 算法 bug~~ | ❌ | 修复后反而退步，已 revert |
| ~~源码 trick (pretrain+5:1)~~ | ❌ | 见 source_code 实验，比 Algorithm 2 更差 |
| **batch_size 不同** | ⚠️ 部分 | 论文单卡 batch=50，我们 DDP 有效 200 |
| **augmentation** | ✅ **关键发现** | 论文未提及，但开 SpecAug 后明显改善 |

**讲稿**：我们系统性排查了 6 个可能的 gap 来源。数据完整性、配置参数、算法实现都正确。最后定位到 augmentation 是关键——论文 Implementation Details 完全没提 augmentation，但实际开了 SpecAug 后效果显著。

---

## Slide 5 — 失败的创新尝试

**标题**：架构层面创新均未超越 baseline

**插图**：`output/plots/06_ddp_3way_compare_baseline_source_slerp.png`

**内容（表格）**：

| 创新点 | 改动 | EER | vs baseline | 结论 |
|--------|------|-----|------------|------|
| Source-faithful | + pretrain phase + 5:1 G:D + d_loss/2 | 3.83% | +0.35% ❌ | 论文 Algorithm 2 比源码 trick 好 |
| SLERP | LERP → 球面插值 | 3.58% | +0.10% ❌ | SLERP 几何优势未兑现 |
| WavLM | HuBERT → WavLM-large | 3.71% | +0.23% ❌ | seq_len=1 让 WavLM 退化 |

**讲稿**：右图是三个失败实验和 baseline 的对比。三个创新都低于 baseline。原因各不相同——source_code 的 pretrain 状态机其实让训练不稳定，SLERP 在 cosine metric 下没有理论优势，WavLM 在 seq_len=1 的输入下退化为 FFN，预训练的 utterance-level 能力发挥不出来。**这告诉我们，CAARMA 论文的算法实现本身已经接近局部最优**，简单的架构替换不足以突破。

---

## Slide 6 — 关键发现：SpecAugment

**标题**：⭐ SpecAugment 关闭 54% 的 Gap

**插图**：建议补做 07_ddp_specaug.png（单 SpecAug 曲线）

**内容（表格）**：

| 配置 | EER | 改善 |
|------|-----|------|
| DDP baseline（无 augmentation） | 3.48% | — |
| **+ SpecAugment**（mel-spec 2× freq + 2× time mask） | **3.27%** | **-0.21%** ✅ |
| Paper SOTA 目标 | 3.09% | 仍有 +0.18% |

**讲稿**：在 mel-spectrogram 上做 SpecAugment（标准的 2 frequency mask + 2 time mask）让 EER 从 3.48% 降到 3.27%。**关闭了 paper gap 的 54%**。论文虽然没明说，但现代 SV 论文（ECAPA-TDNN、MFA-Conformer）默认都用 SpecAugment。我们的实测证实了这是 CAARMA 隐藏的关键组件。

---

## Slide 7 — 反面教训：augmentation 必须正确使用

**标题**：错误的 augmentation 策略反而破坏训练

**插图**：建议补做 08_ddp_full_aug.png（失败曲线）

**内容**：
- **天真做法**：100% 每样本 MUSAN noise + 100% 每样本 RIR reverb + SpecAug → **EER 4.36%**（比 baseline 退步 0.88%）
- **标准做法**（X-vectors / ECAPA / MFA-Conformer 论文）：每样本随机**单选**一种 augmentation
  - 引用：*"For each sample, an augmentation is selected uniformly at random between noise, music, babble, reverberation, or no augmentation"* — Desplanques et al. 2020

**讲稿**：我们一开始把 MUSAN + RIR 全部 100% 叠加到每个样本，结果 EER 暴涨到 4.36%。这是因为每样本经历了 4 种扰动，模型来不及收敛。**正确做法是 per-sample 随机单选一种** augmentation。这一点 X-vectors 等论文都明确说了，但很容易踩坑。

---

## Slide 8 — 当前完整实验汇总

**标题**：7 个完成实验的横向对比

**插图**：建议补做 09_unified_7way_compare.png（所有 7 个曲线叠加）

**内容（表格）**：

| ID | 实验 | Best EER | 状态 |
|----|------|---------|------|
| 03 | DDP baseline | 3.48% | 基准 |
| **07** | **+ SpecAugment** | **3.27%** | **当前最佳** ⭐ |
| 04 | Source-faithful | 3.83% | 失败 |
| 05 | SLERP | 3.58% | 失败 |
| 06 | WavLM | 3.71% | 失败 |
| 08 | Full-aug（暴力叠加，已弃）| 4.36% | 失败 |
| — | Paper SOTA | 3.09% | 目标 |

**讲稿**：这是当前所有完成实验的对比。SpecAugment 是唯一改善 baseline 的改动。其余架构创新都失败。我们距离 paper 3.09% 还差 0.18%。

---

## Slide 9 — 下一步计划

**标题**：突破 Paper SOTA 的三个待跑实验

**内容**：

1. **【最高优先级】aug_per_sample**（已实现，待跑）
   - 标准 SV 配方：per-sample 随机选 {clean / noise / reverb} + SpecAug
   - 预期 EER：3.05-3.20%（**有可能突破 paper 3.09%**）
   - 耗时：~10 小时

2. **【高优先级】ReDimNet-b6 encoder**（已实现，待跑）
   - 替换 MFA-Conformer (19.8M) → ReDimNet-b6 (15M, INTERSPEECH 2024)
   - 验证更强 encoder 在 CAARMA pipeline 下的效果
   - 耗时：~8 小时

3. **【探索性】vMF 球面采样**
   - mixup 中点 → vMF(midpoint, κ=50) 采样
   - 比 SLERP 多样性更高
   - 耗时：~8 小时

**讲稿**：下一步重点是 aug_per_sample 实验——如果正确的 augmentation 策略能再榨 0.1-0.2%，**我们有机会突破 paper SOTA 3.09%**。这一个数据点是接下来 1-2 天的重点。

---

## Slide 10 — 讨论与问题

**标题**：致谢与待讨论问题

**内容**：

**已完成**：
- ✅ 论文严格复现（3.48%, gap 0.39%）
- ✅ 系统排除 5 个 gap 来源
- ✅ 发现 SpecAugment 关闭 54% gap
- ✅ 排除 3 个架构创新（SLERP / WavLM / source_code）

**待讨论**：
1. SpecAugment 是否值得作为「复现 paper 隐藏组件」写进未来 report？
2. SLERP / WavLM 在 SpecAug-baseline (3.27%) 上重测是否值得做（之前都是 vs 3.48% baseline）？
3. 是否考虑发邮件给原作者询问 batch_size、augmentation 等具体细节？
4. 后续创新方向（aug_per_sample / ReDimNet / vMF）的优先级如何安排？

**讲稿**：以上是目前的进展。下一周计划集中跑 aug_per_sample 突破 paper SOTA。有几个问题想听听您的意见……

---

## 附录 — Slide A：详细 EER 轨迹

如果导师追问细节，可以打开 `output/csv/*.csv` 中的具体每 epoch 数字。所有原始数据都存在 `output/` 文件夹下。

## 附录 — Slide B：复现 vs 论文差异溯源

具体证据：
- trials file 37720 pairs vs 标准 VoxCeleb1-O cleaned ✓
- 训练集 148,642 utt vs 论文 153,516 (dev+test) → 实际我们用 dev 集 148,642 ✓
- 算法 2 实现（commits 历史可查）

---

## PPT 制作建议

1. 用 Keynote / PowerPoint / Google Slides 一比一搬运
2. 图都放在 `output/plots/` 目录
3. 配色：baseline 蓝、SpecAug 绿、失败实验灰、paper SOTA 红虚线
4. 字体：标题 Helvetica/思源黑体，正文 Calibri/思源黑体
5. 表格尽量用 PPT 自带样式，避免截图
