# PSC Bridges-2 实验结果归档

平台：PSC Bridges-2，`GPU-shared`，4×V100-32，账号 `cis220031p`。评测协议三作业一致：官方
cleaned VoxCeleb1-O trial 列表（37,611 行）、整段单嵌入、评测集均值中心化、余弦、sklearn
插值 EER、归一化 minDCF（P_target=1e-2 / 1e-3）。代码含修正后的 DDP 全局索引，无官方
snapshot 的 rank 偏移 bug。

> **口径约定**
>
> - 三作业（旧）和五作业（新）使用完全一致的 trial 文件与打分代码，可直接横向比较。
> - 所有 ModelCheckpoint 都按 `cosine_eer`（即 Vox1-O 测试 EER）选模，属于 **test-selected**
>   数值复现口径；报告时需与"开发集选模"严格区分。
> - `precision=16-mixed` 对三个老作业和五个新作业一致，横向可比。
> - `best EER` 列为从日志尾部最近 ~16 个 epoch 可见的最小值；老作业为全球最优，新作业若
>   前期未覆盖则标记 `~`。

## 2026-08-21 批次

| JobID | 实验臂 | 仓库 / commit | 时长 | EER / minDCF(10⁻²) / minDCF(10⁻³)（最终 epoch） | 最优 EER（epoch） |
|---|---|---|---|---|---|
| 44072499 | MLP-D joint-Lsyn | `CAARMA-psc-mlpd361-37d41df` @ 37d41df | 4:38:35 | 3.86 / 0.3227 / 0.5361 | **3.75**（ep16） |
| 44075531 | paper-aligned HuBERT | `caarma_jobs/paper-aligned-ae07e82` @ ae07e82 | 7:54:28 | 3.44 / 0.3211 / 0.4696 | **3.39**（ep24） |
| 44075532 | E3 Natural-Cluster Projection-D | `caarma_jobs/e3-natural-cluster-82c98b2` @ 82c98b2 | 2:54:06 | 3.54 / 0.3492 / 0.4468 | **3.47**（ep19，ep26 持平） |

## 2026-08-22 批次（旧 source-faithful / paper-exact / 主链四臂）

| JobID | 实验臂 | 分支 / commit | 目标 | 时长 | EER / minDCF(10⁻²) / minDCF(10⁻³)（最终 epoch） | 最优 EER |
|---|---|---|---|---|---|---|
| 44189004 | 3.09 source-faithful 复现 | `exp/psc-source-faithful-repro-309` @ bc085f5 | 3.09 | 3:26:11 | 3.51 / 0.3593 / 0.452 | **~3.49**（ep29 前可见） |
| 44192117 | 3.09 paper-exact（无 StepLR 衰减） | `exp/psc-paper-exact-repro-309` @ 1ab649b | 3.09 | 6:28:43 | 3.64 / 0.3946 / 0.4403 | **~3.48**（ep29 前可见） |
| 44190687 | 主链 CRP persistent | `exp/psc-crp-persistent-synth` @ db0d3c2 | 3.57 | 2:58:40 | 3.40 / 0.3618 / 0.5375 | **~3.39**（ep29 前可见） |
| 44191580 | 主链 Concat-D | `exp/psc-concat-crp-persistent-synth` @ 5fdee43 | 3.51 | 2:41:35 | 3.56 / 0.3476 / 0.5093 | **~3.50**（ep29 前可见） |
| 44190693 | 主链 Projection-D | `exp/psc-projection-crp-persistent-synth` @ fbd396a | 3.45 | 2:44:15 | 3.35 / 0.3454 / 0.4019 | **~3.34**（ep29 前可见） |

注：44190692（Concat-D 首次提交）因 `train.py` 硬编码 AutoDL 路径、`--config` 未接入而启动失败，已修
复（`5fdee43`）并由 44191580 替代。

## 2026-08-23 批次（当前运行中）

策略调整：主方法分支按原代码参数复现；3.09 复现只在 **VoxCeleb1** 上进行调参。

### 已运行 / 排队作业

| JobID | 实验臂 | 分支 / commit | 数据 | 状态 | 日志 |
|---|---|---|---|---|---|
| 44235182 | 3.09 repro（调参） | `exp/psc-tuned-repro-309-vox1` @ `3afee15` | Vox1 | COMPLETED | `caarma-tuned-repro-309-44235182.{out,err}` |
| 44255041 | 3.09 source-faithful + am_margin 0.25 / am_scale 32 | `exp/psc-source-faithful-margin025` @ `ff4e77f` | Vox1 | RUNNING | `caarma-vox1-44255041.{out,err}` |
| 44240592 | MLP-D joint-Lsyn dev-trial 选模 | `exp/psc-mlpd361-devsel` @ `?` | Vox1 | COMPLETED | `caarma-mlpd361-devsel-44240592.{out,err}` |
| 44247839 | E0 Xavier control | `exp/psc-e0-xavier-control` @ `243233f` | Vox1 | COMPLETED | `caarma-e0-xavier-44247839.{out,err}` |
| 44247840 | E1 SLERP initialization | `exp/psc-e1-slerp-init` @ `c25c2bf` | Vox1 | COMPLETED | `caarma-e1-slerpinit-44247840.{out,err}` |
| 44247841 | E2 Powered CRP β=0.75 | `exp/psc-e2-powered-crp-beta75` @ `9ad4f2a` | Vox1 | COMPLETED | `caarma-e2-powered75-44247841.{out,err}` |
| 44247842 | Pure Natural-Cluster | `exp/psc-pure-natural-cluster` @ `2e3b0fc` | Vox1 | COMPLETED | `caarma-pure-natclust-44247842.{out,err}` |
| 44247843 | Powered CRP β=0.5 | `exp/psc-powered-crp-beta05` @ `844460d` | Vox1 | COMPLETED | `caarma-powered-beta05-44247843.{out,err}` |
| 44247915 | class-capacity ratio | `exp/psc-class-ratio` @ `372132e` | Vox1 | COMPLETED | `caarma-class-ratio-44247915.{out,err}` |
| 44247936 | sample ratio | `exp/psc-sample-ratio` @ `e3e6a41` | Vox1 | COMPLETED | `caarma-sample-ratio-44247936.{out,err}` |
| 44247937 | Corrected Concat-D | `exp/psc-corrected-concat` @ `8e17e4e` | Vox1 | COMPLETED | `caarma-corrected-concat-44247937.{out,err}` |
| 44247938 | Q-shuffled | `exp/psc-q-shuffled` @ `90b687f` | Vox1 | COMPLETED | `caarma-q-shuffled-44247938.{out,err}` |
| 44247939 | Q-only | `exp/psc-q-only` @ `014e3ad` | Vox1 | COMPLETED | `caarma-q-only-44247939.{out,err}` |
| 44247951 | Cluster-Random-4 | `exp/psc-cluster-random4` @ `1e0cea6` | Vox1 | COMPLETED | `caarma-cluster-random4-44247951.{out,err}` |

### 中期结果（已有 EER 的作业）

| JobID | 实验臂 | 最新可见 EER / minDCF(10⁻²) / minDCF(10⁻³) | 最优 EER |
|---|---|---|---|
| 44235182 | 3.09 repro（调参） | 3.91% / 0.3652 / 0.3917 | **3.91%** |
| 44240592 | MLP-D dev-trial 选模 | 0.92% / 0.1500 / 0.2380 | **0.92%** |

> 44240592 使用 dev-trial 口径，与 Vox1-O 主实验不直接可比。

### 修复后 11 个创新臂的 Epoch 0 结果

第一轮路径修复后的作业已完成 Epoch 0 validation：

| JobID | 实验臂 | Epoch 0 EER / minDCF(10⁻²) / minDCF(10⁻³) |
|---|---|---|
| 44247839 | E0 Xavier control | 10.93% / 0.7242 / 0.8806 |
| 44247840 | E1 SLERP initialization | 10.42% / 0.7452 / 0.7964 |
| 44247841 | E2 Powered CRP β=0.75 | 10.97% / 0.7374 / 0.8337 |
| 44247842 | Pure Natural-Cluster | 10.66% / 0.7324 / 0.8472 |
| 44247843 | Powered CRP β=0.5 | 10.68% / 0.7241 / 0.8466 |
| 44247915 | class-capacity ratio | 7.18% / 0.5622 / 0.7076 |
| 44247936 | sample ratio | 7.53% / 0.5744 / 0.7108 |
| 44247937 | Corrected Concat-D | 7.07% / 0.6039 / 0.7852 |
| 44247938 | Q-shuffled | 7.33% / 0.5741 / 0.6960 |
| 44247939 | Q-only | PENDING（未出 Epoch 0） |
| 44247951 | Cluster-Random-4 | PENDING（未出 Epoch 0） |

> Epoch 0 的 EER 在 10% 左右是正常起点，后续 epoch 会快速下降。

### 当前最新进展（截至本次查询）

| JobID | 实验臂 | 已完成 epoch | 最新 EER / minDCF(10⁻²) / minDCF(10⁻³) |
|---|---|---|---|
| 44235182 | 3.09 repro（调参） | **COMPLETED**（Epoch 29） | **3.70%** / 0.3720 / 0.4251 |
| 44255041 | 3.09 source-faithful + am_margin 0.25 / am_scale 32 | PENDING | — |
| 44240592 | MLP-D dev-trial 选模 | Epoch 20 | **0.19%** / 0.0351 / 0.0843 |
| 44247839 | E0 Xavier control | Epoch 21 | 3.66% / 0.3571 / 0.4857 |
| 44247840 | E1 SLERP initialization | Epoch 20 | 3.49% / 0.3441 / 0.3942 |
| 44247841 | E2 Powered CRP β=0.75 | Epoch 20 | 3.59% / 0.3510 / 0.4259 |
| 44247842 | Pure Natural-Cluster | Epoch 20 | 3.64% / 0.3713 / 0.5128 |
| 44247843 | Powered CRP β=0.5 | Epoch 20 | 3.74% / 0.3644 / 0.5072 |
| 44247915 | class-capacity ratio | Epoch 20 | 3.49% / 0.3441 / 0.4639 |
| 44247936 | sample ratio | Epoch 19 | 3.54% / 0.3714 / 0.4286 |
| 44247937 | Corrected Concat-D | Epoch 19 | 3.61% / 0.3413 / 0.4066 |
| 44247938 | Q-shuffled | Epoch 21 | 3.57% / 0.3230 / 0.3726 |
| 44247939 | Q-only | Epoch 14 | 3.75% / 0.3567 / 0.4027 |
| 44247951 | Cluster-Random-4 | Epoch 2 | 5.85% / 0.5076 / 0.6645 |

> 44240592 使用 dev-trial 口径，与 Vox1-O 主实验不直接可比。

### 2026-08-23 批次最终结果（test-selected best EER）

所有 Vox1-O 实验（除 dev-trial 外）均按 `cosine_eer` 选最优 3 个 checkpoint；下表取 30 epoch 内最低 EER：

| 排名 | JobID | 实验臂 | 最优 EER（epoch） | 对应 minDCF(10⁻²) / minDCF(10⁻³) |
|---|---|---|---|---|
| 1 | 44247936 | sample ratio | **3.44%**（ep20） | 0.3618 / 0.4306 |
| 2 | 44247915 | class-capacity ratio | **3.45%**（ep19） | 0.3583 / 0.4124 |
| 3 | 44247840 | E1 SLERP initialization | **3.48%**（ep25） | 0.3268 / 0.4558 |
| 4 | 44247951 | Cluster-Random-4 | **3.53%**（ep19） | 0.3577 / 0.4177 |
| 5 | 44247939 | Q-only | **3.53%**（ep22） | 0.3629 / 0.4624 |
| 6 | 44247937 | Corrected Concat-D | **3.53%**（ep27） | 0.3466 / 0.4223 |
| 7 | 44247841 | E2 Powered CRP β=0.75 | **3.58%**（ep19） | 0.3591 / 0.4112 |
| 8 | 44247938 | Q-shuffled | **3.57%**（ep21） | 0.3230 / 0.3726 |
| 9 | 44247842 | Pure Natural-Cluster | **3.59%**（ep27） | 0.3643 / 0.5380 |
| 10 | 44247843 | Powered CRP β=0.5 | **3.64%**（ep28） | 0.3696 / 0.4788 |
| 11 | 44247839 | E0 Xavier control | **3.61%**（ep18） | 0.3638 / 0.4457 |
| 12 | 44235182 | 3.09 repro（调参） | **3.65%**（ep24） | 0.3813 / 0.4280 |
| — | 44189004 | 3.09 source-faithful 复现 | **~3.49%** | — |
| — | 44255041 | 3.09 source-faithful + am_margin 0.25 / 32 | **3.58%**（ep23）*进行中* | 0.3502 / 0.4527 |

> 44240592（MLP-D dev-trial 选模）best dev-trial EER 0.15%，但 Vox1-O final 3.91%，
> 说明 dev-trial 口径与 Vox1-O 差距极大，不能混用。

### 已撤销 / 替换

- **44235179–44235181**（CRP persistent / Projection-D / Natural-Cluster Projection-D 重复提交）已撤销。
- **44245125–44245129**（E0 / E1 / E2 / Pure Natural-Cluster / Powered-β0.5 首次）因 `train.py`
  的 `load_config` 缩进错误在 3–6 秒内 FAILED；已修正缩进并重新提交。
- **44245782 / 44245783**（Q-shuffled / Q-only）启动后 FAILED：分支 `train.py` 未解析 `--config`，
  而是硬编码 `/root/autodl-tmp/CAARMA/config.yaml`。已改为 `ArgumentParser` 解析 `--config`。
- **44245744–44245748 / 44245779–44245781 / 44245787 / 44246165 / 44246166**
  在完成 Epoch 0 后的 validation 阶段 FAILED：根因是 `functions/dataset.py` 使用 `self.root + self.paths[idx]`，
  当 root 无末尾 `/` 时路径被错误拼接成 `.../VoxCeleb1/wavidXXXX/...`。已修正为 `os.path.join(...)`。
- **44247149–44247160 / 44247212 / 44247256 / 44247289–44247291 / 44247307**
  同样在完成 Epoch 0 validation 阶段 FAILED：训练数据路径已修复，但 `train.py` 的 `similarity_score`
  仍使用 `self.config['root'] + item[1/2]` 作为 trial key，与 `index_mapping` 中的 `os.path.join` 路径不匹配，
  导致 `KeyError: '.../wavidXXXX/...'`。已把 `train.py` 改为 `os.path.join(self.config['root'], item[1/2])`
  并重新提交为 44247839 起的新批次。

### 3.09 repro 调参锁定配方

```yaml
init_lr: 0.001
discriminator_lr: 0.0002
weight_decay: 1e-8
warmup_step: 2000
lr_scheduler_step_size: 4
lr_scheduler_gamma: 0.5
lambda_adv_mode: fixed
lambda_adv_fixed: 0.05
joint_lsyn_scale: 1.0
am_margin: 0.20
am_scale: 30
batch_size: 50      # per GPU, effective 200 on 4 GPUs
seed: 42
epochs: 30
```

该分支从 `exp/psc-source-faithful-repro-309` 切出，保留了 source-faithful 的 HuBERT-mixup
discriminator 与状态机，但把超参改为 config 驱动，并限制在 VoxCeleb1（`num_spk: 1211`）运行。

## 主要观察

- **论文 3.09 仍未复现**：
  - source-faithful 复现 best ~3.49%，paper-exact 复现 best ~3.48%；
  - 当前 3.09 调参臂（44235182）30 epoch 完整跑完，最终 **3.70%** / 0.3720 / 0.4251，
    与 3.09 仍有 **0.61pp** 差距；
  - paper-exact 去掉 StepLR 后反而比 source-faithful 略差，说明 StepLR 衰减不是阻碍，
    甚至可能有助于收敛。
- **主链趋势成立**：Projection-D（~3.34）< CRP persistent（~3.39）< Concat-D（~3.50）< MLP-D
  baseline（3.75），即 persistent identity → prototype compatibility → 结构化候选的递增效
  益在 PSC 锁定运行时下仍可复现。
- **Projection-D 超过了其历史目标 3.45**；CRP persistent 也超过了 3.57；Concat-D 略逊于 3.51
  但仍在同一区间。
- 所有 5 个 2026-08-22 作业均以 exit 0 完成，无 NaN / walltime 截断。

## 失败/诊断记录

- **44074599**（source-state HuBERT = 原 3.09 复现臂）：FAILED，3 秒，`exit 2:0`。根因为启动器
  `scripts/psc/train_vox1.slurm` 用 `BASH_SOURCE[0]` 自我定位，而 Slurm 执行 spool 副本导致
  `common.sh` 未找到、conda 未激活、`python: command not found`。修复后由 44189004 替代。
- **44190692**（Concat-D 首次）：FAILED，`PermissionError: /root/autodl-tmp/CAARMA/config.yaml`，
  因 `train.py` 的 `cli_main` 仍硬编码 AutoDL 路径且未解析 `--config`。修复后由 44191580 替代。

## VoxCeleb1 + VoxCeleb2 大规模扩展（后续计划）

当前 3.09 repro 只在 VoxCeleb1 上调参。若 Vox1 锁定配方验证有效，再迁移到 Vox1+Vox2 大规模训练。

数据规模：

| 集合 | 说话人 | 语句数 |
|---|---:|---:|
| VoxCeleb1 dev | 1,211 | 148,642 |
| VoxCeleb2 dev | 5,994 | 1,092,009 |
| **合并 dev** | **7,205** | **~1,240,651** |

实施步骤（待执行）：
1. 确认 PSC 数据目录 `/ocean/projects/cis220031p/shared/raw/data/VoxCeleb2/audio/dev/idXXXXX/*.wav`。
2. 使用 `tools/build_voxceleb_csv.py` 生成 `${CAARMA_RUN_ROOT}/manifests/vox1vox2_train.csv`，
   并排除 Vox1-O 的 40 个测试说话人。
3. 配置 `num_spk: 7205`、`discriminator_type: projection`、`batch_size: 50`、`epochs: 30`。
4. 先提交 1-GPU / 1-epoch smoke job 验证形状与无 OOM；通过后提交 4-GPU 全量作业（预计 25–40 小时，
   申请 48:00:00）。
5. 训练完成后在 Vox1-O、Vox1-E、Vox1-H 上离线打分；论文 Table 5 只报告 Vox1-O。

## 运维备注

- `psc_remote.py collect` 默认 `--lines 20000` 会超时；本批使用 `--lines 2000`。
- collect 字段 `best_eer_in_tail` 实际取窗口内最后一次出现，非全局最优；本表已按日志尾部人工复
  核修正。
- 远端日志路径：`/ocean/projects/cis220031p/sge2/logs/caarma-{mlpd361,paper,e3,vox1,pexact,crp,proj,tuned-repro-309}-<JobID>.{out,err}`。

## 归档文件

- 所有 PSC runner collect 的原始 JSON 已同步到 `/Users/shawn/CAARMA/results_psc/`，
  共 60 个文件（含成功、失败、被替换的作业）。主要 JobID 包括：
  - 2026-08-21：`44072499`、`44075531`、`44075532`
  - 2026-08-22：`44189004`、`44192117`、`44190687`、`44191580`、`44190693`
  - 2026-08-23 当前批次：`44235182`、`44240592`、`44247839–44247843`、
    `44247915`、`44247936`、`44247937`、`44247938`、`44247939`、`44247951`
  - 3.09 调参新实验：`44255041`（RUNNING）
  - 被撤销 / 替换的早期尝试：`44235179–44235181`、`44244261`、`44244290`、`44244291`、
    `44244325–44244327`、`44244334–44244335`、`44244374–44244375`、`44244409`、
    `44245125–44245129`、`44245744–44245748`、`44245779–44245783`、`44245787`、
    `44246165`、`44246166`、`44247149–44247151`、`44247159–44247160`、`44247212`、
    `44247256`、`44247289–44247291`、`44247307`。
- 待补充：完整 epoch 曲线 CSV/JSON（目前只有 best / latest 摘要）。
