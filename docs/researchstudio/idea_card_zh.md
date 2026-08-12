# 基于原型条件判别的局部持久虚拟说话人学习

## 研究动机

CAARMA 通过在 embedding 空间混合真实说话人来扩展训练类别，并用对抗训练让合成 embedding 接近真实分布。但一次性生成的
synthetic sample 没有跨 batch 保留的稳定身份；无条件判别器 `D(e)` 也只能判断 embedding 是否像真实分布，不能判断它是否与
分配给自己的真实或合成类别原型一致。此外，全局随机或全局近邻混合仍可能连接当前说话人流形中不兼容的父类。

因此，本工作的核心问题不是“如何生成更多 synthetic embeddings”，而是：如何构造局部有效的虚拟说话人，保留其身份，并反复训练
embedding 与类别原型之间的匹配关系。

## 方法

### 1. 局部兼容的父类候选

令 `W_i` 为真实说话人 `i` 的归一化 AM-Softmax 原型。每个 epoch 使用确定性的 spherical k-means 得到自然分组 `g(i)`，再在同一分组内
选择 cosine 最接近的 `k` 个候选：

$$
\mathcal{N}_i
=
\operatorname{TopK}_{j\ne i,\,g(j)=g(i)}
\cos(W_i,W_j).
$$

cluster 负责给出数据驱动的兼容区域，簇内 cosine top-k 负责保留该区域中局部、容易混淆的父类关系。这与全局 top-k 或簇内均匀随机采样
都不同。

### 2. 持久虚拟说话人身份

每个无序父类对 `{i,j}` 对应一个稳定虚拟类别 `c` 和可学习原型 `W_syn[c]`。CRP-inspired registry 在候选新 pair 与已创建 pair 的复访之间
选择：

$$
P(\mathrm{new}\mid i)
=
\frac{\alpha}{\alpha+\sum_{c\in\mathcal{C}_i} n_c},
\qquad
P(c\mid\mathrm{reuse},i)
=
\frac{n_c}{\sum_{r\in\mathcal{C}_i} n_r}.
$$

这里应称为 CRP-inspired 创建/复用调度器，而不是完整的 Bayesian posterior inference。新类别首次创建时，使用父类的球面中点初始化：

$$
W_{\mathrm{syn}}[c]
\leftarrow
\operatorname{SLERP}(W_i,W_j,\tfrac{1}{2}).
$$

之后每次复访都由同一父类 pair 的不同语音 embedding 产生新的类别样本：

$$
e_{\mathrm{syn}}^{(c)}
=
\operatorname{SLERP}(e_i,e_j,t).
$$

joint AM-Softmax 将该样本分类到同一个持久合成原型，使虚拟说话人不再是只出现一次的扰动点。

### 3. 原型条件下的对抗匹配

判别器接收 embedding 及其 assigned prototype：

$$
q=
\begin{cases}
W_y, & e=e_{\mathrm{real}},\\
W_{\mathrm{syn}}[c], & e=e_{\mathrm{syn}}^{(c)}.
\end{cases}
$$

Projection-D 同时计算边缘真实性和 embedding-prototype compatibility：

$$
D(e,q)
=
h(\phi(e))
+
\frac{\langle\phi(e),\psi(q)\rangle}{\sqrt{d_h}}.
$$

其中内积项是关键机制：它直接表示 `e` 与 `q` 是否匹配。Concat-D 只是拼接 `[e,q]`，需要 MLP 隐式学习二者关系。

### 4. 总体目标与合成比例

$$
\mathcal{L}_{M}
=
\mathcal{L}_{\mathrm{real}}
+\lambda_{\mathrm{syn}}\mathcal{L}_{\mathrm{syn}}
+\lambda_{\mathrm{adv}}\mathcal{L}_{G}.
$$

合成/真实比例定义为

$$
r_{\mathrm{syn}}
=
\frac{N_{\mathrm{synthetic\ samples}}}{N_{\mathrm{real\ samples}}}.
$$

它控制每个 batch 的合成训练曝光量，不等于 registry 中 synthetic classes 与 real classes 的容量比，因此应作为敏感性分析，而不是新的创新点。

## 可证伪实验

在同一运行环境、训练配置和多个随机种子下，依次比较 one-shot、persistent、Concat-D、Projection-D、parent-SLERP 初始化和 cluster-local
top-k。关键观测量为正确与错误条件之间的得分差：

$$
\Delta_q
=
D(e,q^+)-D(e,q^-).
$$

如果机制成立，Projection-D 应比参数量匹配的 Concat-D 获得更大的 `Delta_q` 和更低的 EER。将 `q` 在 real/synthetic bank 内打乱应消除
大部分增益；q-only 不应恢复完整增益；簇内随机四个候选应弱于簇内 cosine top-4。若这些预测在配对种子下不成立，则不能继续主张
compatibility 或局部几何是性能来源。

## 当前证据边界

历史结果与机制方向一致：baseline 3.61，persistent 3.57，Concat-D 3.51，Projection-D 3.45；Projection-D 脱离 persistence 后为
3.72，q-shuffled/q-only 为 3.57/3.59。PPU 上 cluster-local top-4 的最好结果为 3.25，但 AutoDL 对应运行只有 3.52。它们足以提出
研究假设，却不足以直接组成最终主表，原因是环境、checkpoint 选择协议和多个 seed 尚未完全统一。

