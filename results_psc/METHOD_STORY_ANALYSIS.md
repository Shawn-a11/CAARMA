# CAARMA 方法链与下一步调参规划

> 基于 `viz/method_story_html/index.html` 的叙事主线与 `results_psc/` 当前 PSC Bridges-2 实验结果的对照分析。

---

## 1. 方法链主线是否仍然成立？

方法故事（method story）的受控递进为：

```
MLP-D 表示对齐 → SLERP 球面几何 → CRP Persistent 身份
        ↓
   Concat-D 条件 → Projection-D 显式 compatibility
        ↓
Natural Cluster + Local Top-4 候选结构 + class-capacity 调整
```

### 1.1 表示 / 几何 / 身份 / 关系 四层：证据强弱

| 层级 | 关键实验 | 结果 | 是否支持 |
|---|---|---|---|
| 表示 | MLP-D joint-Lsyn baseline | **3.75** best | 基线 |
| 几何 | SLERP init (E1) vs Xavier (E0) | **3.48 vs 3.61** | ✅ 支持 |
| 身份 | CRP persistent vs one-shot | 3.39–3.57 vs 历史 3.72 | ✅ 方向支持 |
| 关系 | Projection-D vs Concat-D | **3.34 vs 3.50** | ✅ 支持 |
| 关系消融 | Q-shuffled / Q-only | 3.57 / 3.53 | ✅ 支持 assignment 重要 |

**结论**：前四层在锁定 runtime 下整体方向成立。

### 1.2 当前数据中的异常点

1. **Concat-D 在主链中不单调**
   - 2026-08-22 主链：CRP 3.39 → Concat-D 3.50 → Projection-D 3.34
   - 期望：CRP 3.57 → Concat-D 3.51 → Projection-D 3.45
   - 实际 Concat-D **差于** CRP，与历史叙事不符。
   - 可能原因：Concat-D 分支的 discriminator 条件实现与 Projection-D 不对称，或 Concat-D 对学习率更敏感。

2. **Cluster-Random-4（3.53）优于 Pure Natural-Cluster（3.59）**
   - 方法故事期望：Natural Cluster 限定区域 → Local Top-4 选局部 parent → 更好。
   - 实际：在限定区域内**随机**选 4 个反而更好。
   - 含义：当前收益主要来自“限制候选在相似 cluster 内”，而非“local top-4 排序”。Local top-4 的增益在历史 PPU 上成立，但在 PSC 锁定 recipe 下未复现。

3. **Powered CRP β  flattening 没有增益**
   - β=1（标准 popularity）历史 ≈3.39；β=0.75 → 3.58；β=0.5 → 3.64。
   - 压平访问权重反而变差，说明当前 registry 的 popularity bias 是有益的。

4. **sample ratio（3.44）与 class-capacity ratio（3.45）成为当前最佳**
   - 这两个都是 exposure / capacity 层面的调参，不在方法故事的核心递进里。
   - 它们超越了所有结构创新（Projection-D 3.34 除外，因为那是 2026-08-22 老数据）。

### 1.3 3.09% 复现状态

| 配置 | best EER | 状态 |
|---|---|---|
| source-faithful | ~3.49 | 差 0.40pp |
| paper-exact | ~3.48 | 差 0.39pp |
| tuned repro (44235182) | 3.65 | 更差 |
| margin0.25/scale32 (44255041) | 3.58 | 未改善 |

**结论**：当前在 VoxCeleb1 上**尚未复现 3.09%**，差距约 0.39–0.56pp。简单调 am_margin / am_scale 不能补齐。

---

## 2. 当前结果排序（Vox1-O，test-selected）

| 排名 | 实验 | best EER | 说明 |
|---:|:---|---:|:---|
| 1 | Main Projection-D (44190693) | **3.34** | 2026-08-22 主链 |
| 2 | Main CRP persistent (44190687) | **3.39** | 2026-08-22 主链 |
| 3 | paper-aligned HuBERT (44075531) | **3.39** | 2026-08-21 |
| 4 | sample ratio (44247936) | **3.44** | 2026-08-23 最佳 |
| 5 | class-capacity ratio (44247915) | **3.45** | 2026-08-23 |
| 6 | E1 SLERP init (44247840) | **3.48** | 消融 |
| 7 | 3.09 paper-exact (44192117) | **3.48** | 复现 |
| 8 | 3.09 source-faithful (44189004) | **3.49** | 复现 |
| 9 | E3 Natural-Cluster Projection-D (44075532) | **3.47** | 候选结构 |
| 10 | Corrected Concat-D (44247937) | **3.53** | 消融 |
| 11 | Q-only (44247939) | **3.53** | 消融 |
| 12 | Cluster-Random-4 (44247951) | **3.53** | 候选控制 |
| 13 | Q-shuffled (44247938) | **3.57** | 消融 |
| 14 | E2 Powered β=0.75 (44247841) | **3.58** | registry |
| 15 | source-faithful margin0.25 (44255041) | **3.58** | 复现调参 |
| 16 | Pure Natural-Cluster (44247842) | **3.59** | 候选 |
| 17 | E0 Xavier (44247839) | **3.61** | 控制 |
| 18 | Powered β=0.5 (44247843) | **3.64** | registry |
| 19 | 3.09 tuned repro (44235182) | **3.65** | 复现调参 |
| 20 | MLP-D baseline (44072499) | **3.75** | 基线 |

> 注：dev-trial 选模实验（44240592）不在此表，因口径不同。

---

## 3. 下一步调参规划

### 3.1 目标优先级

1. **P0：复现 3.09%（差距 0.39pp）**
2. **P1：巩固主方法链的单调性**
3. **P2：把当前最佳 exposure 调参与结构创新组合**

### 3.2 建议实验

#### 实验 A：3.09 复现调参（基于 source-faithful）

当前 best 3.49 来自 source-faithful。下一步控制单一变量：

| 编号 | 变量 | 取值 | 理由 |
|---|---|---|---|
| A1 | `am_scale` | 30 → 28, 32, 36 | 已试 32（3.58），需在 28/36 再确认 |
| A2 | `am_margin` | 0.20 → 0.18, 0.22, 0.25 | 已试 0.25（3.58），0.20 仍最好 |
| A3 | `lambda_adv_fixed` | 0.05 → 0.02, 0.10 | 改变 GAN 强度 |
| A4 | `lr_scheduler_gamma` | 0.5 → 0.7 | 减弱衰减，看是否欠拟合 |
| A5 | `warmup_step` | 2000 → 1000, 4000 | 稳定训练初期 |

建议：先跑 A3 + A4 的组合，因为当前 3.49 已经接近，可能只差一个参数。

#### 实验 B：把 sample ratio / class ratio 嫁接到主链

当前 sample ratio 3.44 是单独调 exposure。应把它与已验证有效的结构组合：

| 编号 | 组合 | 预期 |
|---|---|---|
| B1 | sample ratio + Projection-D | 冲击 <3.34 |
| B2 | sample ratio + CRP persistent + Projection-D | 主链最佳配方 |
| B3 | class ratio R=2 + Projection-D | 扫描最佳 registry 容量 |
| B4 | sample ratio + Natural Cluster + Projection-D | 验证候选结构在好 exposure 下是否生效 |

#### 实验 C：修正 Concat-D 不单调问题

| 编号 | 实验 | 目的 |
|---|---|---|
| C1 | 用 Projection-D 的同一 commit，仅把 critic 替换为 Concat-D | 单变量确认 |
| C2 | 调整 Concat-D 隐藏层 / dropout | 排除实现不稳定 |

#### 实验 D：多 seed 稳定性

对以下 3 个最佳配置各跑 3 个 seed：
- source-faithful（当前复现 best）
- sample ratio
- Projection-D（主链）

每个 seed 只改 `seed`，其余不动。用于报告 mean±std。

### 3.3 暂时不要做的事情

- **不要现在上 Vox1+Vox2**：3.09 在 Vox1 上都没复现，迁移到大数据集会引入额外变量。
- **不要继续 flatten CRP**：β 实验已明确 popularity bias 有益。
- **不要继续 Local Top-4 单独优化**：在 PSC recipe 下未超过 Cluster-Random-4，需先与 exposure 调参组合后再评估。

---

## 4. 逻辑链叙事修正建议

如果最终数据保持当前趋势，论文叙事需要微调：

1. **把“sample ratio / class ratio”纳入方法链的尾部**：它们不是独立模块，但说明 persistent registry 的 exposure 与容量对性能有实质影响。
2. **弱化 Concat-D 的单独贡献**：当前数据下 Concat-D 不如 CRP persistent，可把它写成“通向 Projection-D 的中间步骤”而非独立改进。
3. **把 Natural Cluster 的卖点从“local top-4”改为“限定候选区域”**：Cluster-Random-4 vs Pure Natural-Cluster 的结果支持这一点。
4. **Projection-D 仍是最强单一模块**：在所有受控比较中保持最优，应作为关系建模的核心证据。

---

## 5. 近期行动清单

- [ ] 跑 A3（lambda_adv）+ A4（lr gamma）复现 3.09
- [ ] 跑 B1（sample ratio + Projection-D）
- [ ] 跑 B3（class ratio R=2 + Projection-D）
- [ ] 跑 C1（单变量 Concat-D 重测）
- [ ] 对 3 个最佳配置跑多 seed
- [ ] 更新 method story PPT/网页中的证据数字与叙事
