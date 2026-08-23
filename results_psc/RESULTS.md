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
| 44235182 | 3.09 repro（调参） | `exp/psc-tuned-repro-309-vox1` @ `3afee15` | Vox1 | RUNNING (01:14) | `caarma-tuned-repro-309-44235182.{out,err}` |
| 44240592 | MLP-D joint-Lsyn dev-trial 选模 | `exp/psc-mlpd361-devsel` @ `?` | Vox1 | RUNNING (01:10) | `caarma-mlpd361-devsel-44240592.{out,err}` |
| 44245744 | E0 Xavier control | `exp/psc-e0-xavier-control` @ `233dc18` | Vox1 | RUNNING | `caarma-e0-xavier-44245744.{out,err}` |
| 44245745 | E1 SLERP initialization | `exp/psc-e1-slerp-init` @ `092cad5` | Vox1 | RUNNING | `caarma-e1-slerpinit-44245745.{out,err}` |
| 44245746 | E2 Powered CRP β=0.75 | `exp/psc-e2-powered-crp-beta75` @ `0b7a44c` | Vox1 | RUNNING | `caarma-e2-powered75-44245746.{out,err}` |
| 44245747 | Pure Natural-Cluster | `exp/psc-pure-natural-cluster` @ `7bb5cad` | Vox1 | RUNNING | `caarma-pure-natclust-44245747.{out,err}` |
| 44245748 | Powered CRP β=0.5 | `exp/psc-powered-crp-beta05` @ `b46b696` | Vox1 | RUNNING | `caarma-powered-beta05-44245748.{out,err}` |
| 44245779 | class-capacity ratio | `exp/psc-class-ratio` @ `fe7120c` | Vox1 | PENDING | `caarma-class-ratio-44245779.{out,err}` |
| 44245780 | sample ratio | `exp/psc-sample-ratio` @ `b8fb936` | Vox1 | PENDING | `caarma-sample-ratio-44245780.{out,err}` |
| 44245781 | Corrected Concat-D | `exp/psc-corrected-concat` @ `e8c7349` | Vox1 | PENDING | `caarma-corrected-concat-44245781.{out,err}` |
| 44246165 | Q-shuffled | `exp/psc-q-shuffled` @ `47664f8` | Vox1 | PENDING | `caarma-q-shuffled-44246165.{out,err}` |
| 44246166 | Q-only | `exp/psc-q-only` @ `3d18eb4` | Vox1 | PENDING | `caarma-q-only-44246166.{out,err}` |
| 44245787 | Cluster-Random-4 | `exp/psc-cluster-random4` @ `947021d` | Vox1 | PENDING | `caarma-cluster-random4-44245787.{out,err}` |

### 已撤销 / 替换

- **44235179–44235181**（CRP persistent / Projection-D / Natural-Cluster Projection-D 重复提交）已撤销，
  相关分支的修复版本见上表 44245744 起的新批次。
- **44245125–44245129**（E0 / E1 / E2 / Pure Natural-Cluster / Powered-β0.5 首次修复提交）因 `train.py`
  的 `load_config` 缩进错误在 3–6 秒内 FAILED；已修正缩进并重新提交为 44245744–44245748。
- **44245782 / 44245783**（Q-shuffled / Q-only）启动后 FAILED：分支 `train.py` 未解析 `--config`，
  而是硬编码 `/root/autodl-tmp/CAARMA/config.yaml`。已改为 `ArgumentParser` 解析 `--config`，
  重新提交为 44246165 / 44246166。

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

- **论文 3.09 未复现**：两个 3.09 臂（source-faithful 3.51、paper-exact 3.48-best）与 3.09 仍有
  ~0.4pp 差距；paper-exact 去掉 StepLR 后反而比 source-faithful 略差，说明 StepLR 衰减不是阻
  碍，甚至可能有助于收敛。
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

- JSON 原始 collect 输出：`44072499.json`、`44075531.json`、`44075532.json`、
  `44189004.json`、`44192117.json`、`44190687.json`、`44191580.json`、`44190693.json`。
- 当前运行中作业的 collect JSON 待生成。
- 待补充：完整 epoch 曲线 CSV/JSON（作业结束后拉取日志生成）。
