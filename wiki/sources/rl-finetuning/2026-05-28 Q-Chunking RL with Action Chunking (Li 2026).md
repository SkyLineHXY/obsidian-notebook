---
type: source
tags: [Reinforcement Learning, Action Chunking, Offline-to-Online RL, Flow Matching, Q-Learning, OGBench]
sources: [raw/assets/papers/RL(Reinforce Learning)/Li 等 - 2026 - Reinforcement Learning with Action Chunking.pdf]
created: 2026-05-28
updated: 2026-05-28
---

# Reinforcement Learning with Action Chunking

**arXiv**: 2506.XXXXX（待确认）
**作者**: Qiyang Li, Zhiyuan Zhou, Sergey Levine
**机构**: UC Berkeley
**发表**: NeurIPS 2026
**代码**: https://github.com/ColinQiyangLi/qc
**摄取日期**: 2026-05-28
**摄取来源**: 用户添加 PDF + MinerU 转换

[[raw/assets/papers/RL(Reinforce Learning)/Li 等 - 2026 - Reinforcement Learning with Action Chunking.pdf]]
[[raw/sources/papers/RL(Reinforce Learning)/Li 等 - 2026 - Reinforcement Learning with Action Chunking/Li 等 - 2026 - Reinforcement Learning with Action Chunking.md]]

---

## 一句话摘要

Q-chunking 将动作分块（action chunking）引入 TD-based RL，同时解决离线转在线 RL 中的两大核心障碍——长视野 value backup 慢和在线探索缺乏时间连贯性——在 OGBench 上整体成功率从 52% 提升到 86%，大幅超越所有先前方法。

---

## 核心背景与动机

### 问题：离线转在线 RL 的探索困境

**离线转在线 RL 设置**：有一个离线先验数据集 $\mathcal{D}$，目标是最大化在线采样效率。两大核心挑战：

1. **Long-horizon value backup 慢**：标准 1-step TD 每次只向前传播一步价值，有效视野 $\tilde{H} = 1/(1-\gamma)$ 越大，学习越慢。n-step return 能加速，但存在 off-policy bias。

2. **在线探索缺乏时间连贯性**：随机动作导致采集的轨迹局限在初始状态附近，无法有效覆盖任务空间。现有技能分层方法（HRL）存在双层优化不稳定的问题。

### 关键洞察

Action chunking（在模仿学习中因 [[wiki/concepts/imitation-learning/ACT]] 而普及）在 RL 中有两个被忽视的好处：
1. **Unbiased n-step backup**：将 Q 函数定义在动作序列上，使 h-step backup 与数据收集策略无关，消除偏差。
2. **时间连贯探索**：行为约束作用于动作序列，自然提取离线数据中的时序技能，探索时覆盖更广的状态空间。

---

## 方法：Q-Chunking

![[raw/sources/papers/RL(Reinforce Learning)/Li 等 - 2026 - Reinforcement Learning with Action Chunking/images/972fe01de55317cdf35c0d17ae973b4759ea956ca791f47befda3bd4540f4dbd.jpg]]
*Figure 1：Q-chunking 方案总览。左：在时间扩展动作空间上运行 actor-critic，(1) 高效 value backup，(2) 通过时间连贯动作有效探索。右：QC 在 OGBench 5 个域上的聚合表现，离线 1M 步后在线 1M 步。*

### 核心设计：时间扩展动作空间

标准 actor-critic 学习 $Q(s_t, a_t)$ 和 $\pi(a_t \mid s_t)$；Q-chunking 改为学习：

$$
\text{Q-Chunking Policy:}\quad \pi_\psi(\boldsymbol{a}_{t:t+h} \mid s_t) := \pi_\psi(a_t, a_{t+1}, \dots, a_{t+h-1} \mid s_t)
$$

$$
\text{Q-Chunking Critic:}\quad Q_\theta(s_t, \boldsymbol{a}_{t:t+h}) := Q_\theta(s_t, a_t, a_{t+1}, \dots, a_{t+h-1})
$$

**TD 损失**（h 步 backup，无偏）：

$$
L(\theta) = \mathbb{E}\left[\left(Q_\theta(s_t, \boldsymbol{a}_{t:t+h}) - \underbrace{\sum_{t'=0}^{h-1}\gamma^{t'} r_{t+t'}}_{\text{unbiased}} - \gamma^h Q_{\bar\theta}(s_{t+h}, \boldsymbol{a}_{t+h:t+2h})\right)^2\right] \tag{4}
$$

**为什么无偏？** 标准 n-step return 的 bias 来自 bootstrap 时动作 $a_{t+n}$ 是当前策略采样而非数据动作；Q-chunking 的 Q 函数输入包含数据中的完整动作序列 $\boldsymbol{a}_{t:t+h}$，bootstrap 时也使用同一长度的动作块，因此 bias 为零（见 Theorem A.1）。

三种 backup 对比：

$$
Q(s_t, a_t) \leftarrow r_t + \gamma Q(s_{t+1}, a_{t+1}) \quad \text{(1-step TD)} \tag{5}
$$

$$
Q(s_t, a_t) \leftarrow \underbrace{\sum_{t'=t}^{t+h-1}\gamma^{t'-t}r_{t'}}_{\text{biased}} + \gamma^h Q(s_{t+h}, a_{t+h}) \quad \text{(n-step return)} \tag{6}
$$

$$
Q(s_t, \boldsymbol{a}_{t:t+h}) \leftarrow \underbrace{\sum_{t'=t}^{t+h-1}\gamma^{t'-t}r_{t'}}_{\text{unbiased}} + \gamma^h Q(s_{t+h}, \boldsymbol{a}_{t+h:t+2h}) \quad \text{(Q-chunking)} \tag{7}
$$

### 行为约束：时间连贯探索

策略优化目标（含约束）：

$$
L(\psi) = -\mathbb{E}_{\boldsymbol{a}_{t:t+h} \sim \pi_\psi}\left[Q_\theta(s_t, \boldsymbol{a}_{t:t+h})\right], \quad \text{s.t.} \; D(\pi_\psi(\boldsymbol{a}_{t:t+h}\mid s_t),\, \pi_\beta(\boldsymbol{a}_{t:t+h}\mid s_t)) \leq \varepsilon \tag{8}
$$

其中 $\pi_\beta$ 为离线数据的行为分布，通过 flow-matching 建模（Gaussian policy 效果差，如 Figure 2 所示）。

### 两种实例化算法

**QC（隐式 KL 约束，Best-of-N 采样）**：
- 从行为流策略 $f_\xi(\cdot\mid s)$ 采样 N 个动作块：$\{\boldsymbol{a}^1, \dots, \boldsymbol{a}^N\} \sim f_\xi(\cdot\mid s)$
- 选 Q 值最大者：$\boldsymbol{a}^\star \leftarrow \arg\max_{\boldsymbol{a}} Q(s, \boldsymbol{a})$
- KL 上界（来自 prior work）：$D_{\mathrm{KL}}(\boldsymbol{a}^\star \| f_\xi) \leq \log N - \frac{N-1}{N}$
- 无需单独参数化策略网络，$f_\xi$ 即为策略

**QC-FQL（显式 $W_2$ 约束，单步蒸馏）**：
- 噪声条件策略 $\mu_\psi(s, z)$ 从 Gaussian 噪声一步输出动作块
- 损失同时最大化 Q 值并 BC 蒸馏行为流策略（$W_2$ 上界）：

$$
L(\psi) = \mathbb{E}\left[\alpha\|\boldsymbol{z}^1 - \mu_\psi(s_t, \boldsymbol{z}^0)\|_2^2 - Q(s_t, \mu_\psi(s_t, \boldsymbol{z}))\right] \tag{13}
$$

---

## 实验结果

**基准**：OGBench（5 个域，25 个任务：puzzle-3x3-sparse / scene-sparse / cube-double/triple/quadruple）+ Robomimic（3 个任务，多人类数据集）。

**主要结果（Table 1）**：离线→在线整体成功率（%）：

| 方法 | 离线 | 在线 |
|------|------|------|
| FQL | 37 | 58 |
| BFN（强基线）| 51 | 63 |
| **QC-FQL（ours）** | **38** | **86** |
| **QC（ours）** | **52** | **86** |

- 在最难的 cube-quadruple 域：QC 在线 73%，QC-FQL 77%，BFN 仅 12%
- OGBench 和 Robomimic 上，所有 Q-chunking 变体均优于对应的 1-step TD 和 n-step return 基线

**消融分析**：
- 动作块长度 $h$：$h=10$ 峰值最优，$h=5$ 性价比最高（默认）；$h=50$ 完全失败
- Critic ensemble $K=10$ 比 $K=2$ 更好，但代价更高
- 增大 UTD ratio 对 QC 无显著提升

**探索可视化**：QC 的末端执行器轨迹在训练早期覆盖更广、暂停更少，时间连贯度量（相邻 5 步末端位移均值）显著高于 BFN。

---

## 局限性与未来工作

1. 固定块大小：chunk length 需任务相关的超参调优，无自适应机制
2. 高频控制场景下，开环动作序列会牺牲反馈响应能力
3. 行为约束仅近似，不能保证严格的分布约束

---

## 与已有方法的关系

- **[[wiki/sources/rl-finetuning/2026-05-25 Decoupled Q-Chunking (Li 2025)]]**：同一作者（Qiyang Li）的相关工作。Li 2025 在此 Q-chunking recipe 基础上进一步提出 Partial Critic Distillation（DQC），解耦 critic/policy 的动作块大小。Li 2026 是 Q-chunking 的通用 recipe 论文。
- **[[wiki/concepts/imitation-learning/ACT]]**：Action chunking 在 IL 中的来源，本文将其引入 RL。
- **[[wiki/concepts/benchmarks/OGBench]]**：本文在 OGBench 全部 5 个域（25 任务）上评测，是继 #47/#48 后第三篇主要使用 OGBench 的论文。
- **[[wiki/concepts/generative-models/Flow Matching]]**：行为策略采用 flow-matching 建模，Gaussian 策略在动作分块 RL 中效果差。

---

## 新概念追踪

**已达 ≥2 来源，可升级为 Concept 页**：
- **Action Chunking in RL（Q-chunking）**：来源 #48（Li 2025 DQC）+ **来源 #49（本文）**，均形式化了动作分块 critic 的无偏 value backup 理论。建议新建 `wiki/concepts/rl/Action Chunking Q-Learning.md`。

**首次出现，追踪中**：
- **Best-of-N 采样作为隐式 KL 约束**：从行为策略采 N 个候选取 Q 最大者，KL 上界 $\leq \log N - (N-1)/N$；仅来源 49
- **时间连贯度量（Temporal Coherency Metric）**：每 5 步末端执行器位移均值作为探索质量量化指标；仅来源 49

---

## 关联页面

- [[wiki/concepts/rl/Action Chunking Q-Learning]] — 待建（≥2 来源阈值已达）
- [[wiki/concepts/benchmarks/OGBench]] — 主要评测基准
- [[wiki/concepts/imitation-learning/ACT]] — Action chunking 在 IL 的来源
- [[wiki/sources/rl-finetuning/2026-05-25 Decoupled Q-Chunking (Li 2025)]] — 同作者后续工作
- [[wiki/analyses/RL微调生成模型的技术挑战与解决方案]] — 离线 Q-learning 路线背景
