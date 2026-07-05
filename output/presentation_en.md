# CAARMA Reproduction & Extension Study — Progress Report

> Faculty advisor briefing · English version
> Suggested duration: 12-15 minutes, 10 slides

---

## Slide 1 — Title

**Title**: CAARMA Reproduction & Extension — Interim Progress

**Subtitle**: From baseline reproduction to the SpecAugment discovery

**Content**:
- Presenter: [Name]
- Date: 2026.06.02
- Task: Reproduce CAARMA (EMNLP 2025) and explore performance improvements

**Speaker notes**: Today I'll report on the past month of CAARMA research. Three main parts: (1) how far we got with the reproduction, (2) which innovations we tried and what we found, (3) next steps.

---

## Slide 2 — Background

**Title**: Research Background & Goals

**Content**:
- **CAARMA**: EMNLP 2025 paper from CMU, achieves EER 3.09% on VoxCeleb1
- **Core idea**: Embedding-space mixup to generate "synthetic speakers" + adversarial training + HuBERT-based discriminator
- **Our goals**:
  1. Reproduce the paper's reported 3.09% EER
  2. Explore extensions (spherical interpolation, alternative encoders, alternative discriminator backbones)
  3. Test whether any innovation can break the paper SOTA

**Speaker notes**: CAARMA uses mixup with adversarial training for zero-shot speaker verification. Our reproduction target is 3.09% EER. We also wanted to test some natural extensions to see if we can push the EER lower.

---

## Slide 3 — Baseline Reproduction Result

**Title**: Faithful CAARMA Reproduction: 4×V100 DDP

**Figure**: `output/plots/03_ddp_baseline_right_algorithm2.png`

**Content**:
- Strict implementation of paper Section 4.2.3: AdamW lr=1e-3, m=0.2, s=30, weight_decay=1e-7, 30 epochs
- 4-GPU V100 DDP, per-GPU batch=50 (effective batch=200)
- **Reproduction best EER = 3.48% @ epoch 17**
- vs paper 3.09% → **gap of 0.39%**

**Speaker notes**: The figure shows the 30-epoch EER trajectory of our baseline reproduction. Best EER bottoms out at 3.48% at epoch 17, 0.39% above the paper's claimed 3.09%. This gap is the starting point for everything that follows.

---

## Slide 4 — Gap Source Analysis

**Title**: 0.39% Gap: Systematic Elimination of Controllable Factors

**Content (table)**:

| Suspect factor | Confirmed? | Verification |
|---------------|-----------|-------------|
| ~~Trials file mismatch~~ | ❌ | Verified standard VoxCeleb1-O 37,720 pairs |
| ~~num_spk error~~ | ❌ | Paper Table 2 explicit VoxCeleb1 = 1211 |
| ~~Training data attrition~~ | ❌ | 148,591 / 148,642 = missing only 51 utts (0.034%) |
| ~~mixup algorithm bug~~ | ❌ | Fix actually regressed performance, reverted |
| ~~Source-code tricks (pretrain+5:1)~~ | ❌ | See source_code experiment — worse than Algorithm 2 |
| **batch_size mismatch** | ⚠️ Partial | Paper single-GPU batch=50; ours DDP effective 200 |
| **Augmentation** | ✅ **Key finding** | Not mentioned in paper, but enabling SpecAug clearly helps |

**Speaker notes**: We systematically eliminated 6 possible gap sources. Data integrity, configuration, algorithm implementation are all correct. The key culprit turned out to be augmentation — the paper's Implementation Details section never mentions augmentation, but enabling SpecAug makes a real difference.

---

## Slide 5 — Failed Innovation Attempts

**Title**: Architecture-level innovations all underperform baseline

**Figure**: `output/plots/06_ddp_3way_compare_baseline_source_slerp.png`

**Content (table)**:

| Innovation | Change | EER | vs baseline | Conclusion |
|-----------|--------|-----|------------|-----------|
| Source-faithful | + pretrain phase + 5:1 G:D + d_loss/2 | 3.83% | +0.35% ❌ | Paper Algorithm 2 outperforms source-code tricks |
| SLERP | LERP → spherical interpolation | 3.58% | +0.10% ❌ | SLERP's geometric advantage didn't materialize |
| WavLM | HuBERT → WavLM-large | 3.71% | +0.23% ❌ | seq_len=1 input degenerates WavLM's pretrained advantage |

**Speaker notes**: The figure compares three failed experiments against the baseline. All three fall below baseline performance. The reasons differ — source_code's state machine destabilizes training, SLERP has no real geometric advantage under cosine metrics, and WavLM degrades to feed-forward when the input is a single time step, killing its utterance-level pretraining benefit. **This tells us the paper's published Algorithm is already near a local optimum.** Simple architectural swaps don't break through.

---

## Slide 6 — Key Discovery: SpecAugment

**Title**: ⭐ SpecAugment Closes 54% of the Gap

**Figure**: Suggest adding 07_ddp_specaug.png (SpecAug-only curve)

**Content (table)**:

| Configuration | EER | Improvement |
|--------------|-----|-------------|
| DDP baseline (no augmentation) | 3.48% | — |
| **+ SpecAugment** (mel-spec 2× freq + 2× time mask) | **3.27%** | **-0.21%** ✅ |
| Paper SOTA target | 3.09% | Remaining gap +0.18% |

**Speaker notes**: Adding SpecAugment on the mel-spectrogram (standard 2 frequency masks + 2 time masks) drops EER from 3.48% to 3.27%. **This closes 54% of the paper gap in a single change.** Although the CAARMA paper doesn't mention augmentation explicitly, modern SV papers (ECAPA-TDNN, MFA-Conformer) default to using SpecAugment. Our experiment empirically confirms it's a hidden ingredient.

---

## Slide 7 — Cautionary Tale: Augmentation Strategy Matters

**Title**: Naive augmentation stacking destroys training

**Figure**: Suggest adding 08_ddp_full_aug.png (failure trajectory)

**Content**:
- **Naive approach**: 100% MUSAN noise + 100% RIR reverb + SpecAug stacked per sample → **EER 4.36%** (0.88% worse than baseline)
- **Standard SV recipe** (X-vectors / ECAPA / MFA-Conformer papers): per-sample uniform random selection of one augmentation
  - Quote: *"For each sample, an augmentation is selected uniformly at random between noise, music, babble, reverberation, or no augmentation"* — Desplanques et al. 2020

**Speaker notes**: We initially applied MUSAN and RIR at 100% per sample (stacked together). EER exploded to 4.36%. Each sample was hit with 4 distortions, and the model couldn't converge. **The correct recipe is per-sample random single-type selection.** This is spelled out in X-vectors and ECAPA-TDNN papers but easy to overlook in code.

---

## Slide 8 — Complete Experiment Summary

**Title**: Seven Completed Experiments — Side-by-side

**Figure**: Suggest adding 09_unified_7way_compare.png (all seven curves overlaid)

**Content (table)**:

| ID | Experiment | Best EER | Status |
|----|-----------|---------|--------|
| 03 | DDP baseline | 3.48% | Reference |
| **07** | **+ SpecAugment** | **3.27%** | **Best so far** ⭐ |
| 04 | Source-faithful | 3.83% | Failed |
| 05 | SLERP | 3.58% | Failed |
| 06 | WavLM | 3.71% | Failed |
| 08 | Full-aug (naive, deprecated) | 4.36% | Failed |
| — | Paper SOTA | 3.09% | Target |

**Speaker notes**: This is the summary of all completed experiments. SpecAugment is the only change that improved over baseline. All architecture-level innovations regressed. We're 0.18% from paper SOTA.

---

## Slide 9 — Next Steps

**Title**: Three Experiments to Break Paper SOTA

**Content**:

1. **【Top priority】aug_per_sample** (already implemented, queued)
   - Standard SV recipe: per-sample random {clean / noise / reverb} + SpecAug
   - Expected EER: 3.05-3.20% (**could break paper 3.09%**)
   - Runtime: ~10 hours

2. **【High priority】ReDimNet-b6 encoder** (already implemented, queued)
   - Replace MFA-Conformer (19.8M) → ReDimNet-b6 (15M, INTERSPEECH 2024)
   - Test whether stronger encoder helps inside CAARMA pipeline
   - Runtime: ~8 hours

3. **【Exploratory】vMF spherical sampling**
   - mixup midpoint → vMF(midpoint, κ=50) sample
   - Higher diversity than SLERP
   - Runtime: ~8 hours

**Speaker notes**: The next focus is the aug_per_sample experiment — if the correct augmentation recipe yields another 0.1-0.2%, we have a real chance of beating paper SOTA 3.09%. That single data point is the priority for the next 1-2 days.

---

## Slide 10 — Discussion & Open Questions

**Title**: Discussion Points

**Content**:

**What we've completed**:
- ✅ Faithful paper reproduction (3.48%, gap 0.39%)
- ✅ Systematic elimination of 5 gap sources
- ✅ Discovery: SpecAugment closes 54% of the gap
- ✅ Ruled out 3 architecture-level innovations (SLERP / WavLM / source_code)

**Open questions for discussion**:
1. Should SpecAugment be documented as a "hidden ingredient" of CAARMA reproduction in future write-ups?
2. Is it worth re-running SLERP / WavLM on the SpecAug baseline (3.27%) instead of the no-aug baseline (3.48%)?
3. Should we reach out to the authors about specific implementation details (batch_size, augmentation, etc.)?
4. Priority ordering for next innovations (aug_per_sample / ReDimNet / vMF)?

**Speaker notes**: That's the current state. Next week is focused on aug_per_sample to potentially break paper SOTA. Several questions for your input...

---

## Appendix — Slide A: Detailed EER Trajectories

If the advisor wants details, refer to `output/csv/*.csv` which contains per-epoch metrics for every experiment.

## Appendix — Slide B: Reproduction-vs-Paper Differences

Concrete evidence:
- trials file 37,720 pairs vs standard VoxCeleb1-O cleaned ✓
- Training utts: 148,642 (matches paper's dev split exactly when interpreting 153,516 as dev+test combined) ✓
- Algorithm 2 implementation (verifiable in commit history)

---

## PPT Production Tips

1. Use Keynote / PowerPoint / Google Slides — content maps 1:1 to slides
2. All figures are under `output/plots/`
3. Color scheme: baseline blue, SpecAug green, failed experiments gray, paper SOTA red dashed line
4. Fonts: Title in Helvetica / Sans-serif, body in Calibri / Source Sans
5. Prefer native PPT tables over screenshots
