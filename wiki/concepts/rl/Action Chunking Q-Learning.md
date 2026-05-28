---
type: concept
tags: [Reinforcement Learning, Action Chunking, Q-Learning, Offline-to-Online RL, TD Learning]
sources: [来源 48（Li 2025 DQC）, 来源 49（Li 2026 Q-chunking）]
created: 2026-05-28
updated: 2026-05-28
---

# Action Chunking Q-Learning

动作分块（Action Chunking）在 TD-based RL 中的应用，通过将策略与 Critic 的动作空间从单步扩展到 h 步动作序列，同时解决长视野 value backup 慢和在线探索缺乏时间连贯性两大核心困难。

---

## 核心洞察

在全观测 MDP 中，最优策略是 Markovian 的，因此 action chunking 在 RL 中显得"不必要"。然而：

1. **探索阶段**：即使最终策略是 Markovian 的，探索时使用时间连贯的非 Markovian 技能更高效
2. **离线数据**：人类 tele-operation 或脚本策略数据固有非 Markovian 结构，动作块级行为约束能更好捕获此分布
3. **无偏 n-step backup**：将 Q 函数定义在动作序列上可消除传统 n-step return 的 off-policy bias

---

## 严谨数学推导

### 符号定义

- MDP $(\mathcal{S}, \mathcal{A}, \rho, T, r, \gamma)$，有效视野 $\tilde{H} = 1/(1-\gamma)$
- 动作块大小 $h$（chunk length）
- 扩展动作空间：$\boldsymbol{a}_{t:t+h} = (a_t, a_{t+1},\dots,a_{t+h-1}) \in \mathcal{A}^h$
- Q-chunking 策略：$\pi_\psi(\boldsymbol{a}_{t:t+h}\mid s_t)$
- Q-chunking Critic：$Q_\theta(s_t, \boldsymbol{a}_{t:t+h})$

### 三种 backup 对比

**标准 1-step TD**（每步传播 1 步）：

$$
Q(s_t, a_t) \leftarrow r_t + \gamma\, Q(s_{t+1}, a_{t+1}) \tag{5}
$$

**n-step return**（加速但有偏）：

$$
Q(s_t, a_t) \leftarrow \underbrace{\sum_{t'=t}^{t+h-1}\gamma^{t'-t}r_{t'}}_{\text{biased}} + \gamma^h\, Q(s_{t+h}, a_{t+h}), \quad a_{t+h}\sim\pi_\psi(\cdot\mid s_{t+h}) \tag{6}
$$

**偏差来源**：bootstrap 时 $a_{t+h}$ 来自当前策略 $\pi_\psi$，而数据中的 $r_{t:t+h}$ 是其他策略产生的，二者分布不匹配。

**Q-chunking backup**（加速且无偏）：

$$
Q(s_t, \boldsymbol{a}_{t:t+h}) \leftarrow \underbrace{\sum_{t'=t}^{t+h-1}\gamma^{t'-t}r_{t'}}_{\text{unbiased}} + \gamma^h\, Q(s_{t+h}, \boldsymbol{a}_{t+h:t+2h}), \quad \boldsymbol{a}_{t+h:t+2h}\sim\pi_\psi(\cdot\mid s_{t+h}) \tag{7}
$$

**无偏证明关键**：Q-chunking 的 Critic 输入包含完整动作序列 $\boldsymbol{a}_{t:t+h}$；bootstrap 时 $\boldsymbol{a}_{t+h:t+2h}$ 是从当前策略采样的新的完整动作块，与数据中产生 $r_{t:t+h}$ 的动作序列 $\boldsymbol{a}_{t:t+h}$ 相互独立，不存在分布错配。（形式化见 Li 2026 Theorem A.1）

### 行为约束目标

在扩展动作空间上加行为约束：

$$
\max_{\pi_\psi}\; \mathbb{E}_{\boldsymbol{a}\sim\pi_\psi}\left[Q_\theta(s_t, \boldsymbol{a}_{t:t+h})\right] \quad \text{s.t.}\; D(\pi_\psi(\boldsymbol{a}_{t:t+h}\mid s_t),\, \pi_\beta(\boldsymbol{a}_{t:t+h}\mid s_t)) \leq \varepsilon \tag{8}
$$

$\pi_\beta$ 为离线数据行为分布（用 flow-matching 近似）。约束作用于动作序列级，使策略从离线数据中提取时间连贯的技能。

### Value Bias 上界（来自 Li 2025 DQC）

Li 2025（Decoupled Q-Chunking）形式化了动作块大小对 value bias 的影响上界：

$$
\text{Value Bias} \leq \mathcal{O}(\varepsilon_h H \bar{H})
$$

其中 $\varepsilon_h$ 为在 chunk 级 TD 误差，$H$ 为时间视野，$\bar{H}$ 为 chunk 大小，表明适中的 chunk 大小能在加速 backup 与控制 bias 之间取得平衡。

---

## 实现变体

| 方法 | 来源 | Critic/Policy 关系 | 行为约束 |
|------|------|--------------------|----------|
| **QC**（best-of-N）| Li 2026 #49 | Critic 和 Policy 均在 h-chunk 空间 | 隐式 KL：$D_{\mathrm{KL}} \leq \log N - (N-1)/N$ |
| **QC-FQL**（W2 蒸馏）| Li 2026 #49 | 单步噪声 $\mu_\psi$ 蒸馏行为 flow | 显式 $W_2$ 上界 BC 损失 |
| **DQC**（解耦分块）| Li 2025 #48 | Critic 用大 chunk，Policy 用小 chunk | Partial Critic Distillation（expectile regression） |

**DQC 的额外贡献**（来自 [[wiki/sources/rl-finetuning/2026-05-25 Decoupled Q-Chunking (Li 2025)]]）：通过 expectile regression 将大块 Critic 蒸馏为小块 partial Critic，解耦 Critic 和 Policy 的 chunk 大小，兼顾 value backup 效率与策略的精细控制。

---

## 实验证据

### OGBench 结果（Li 2026 Table 1）

| 方法 | 离线 | 在线 | 优势 |
|------|------|------|------|
| FQL（1-step TD + flow）| 37% | 58% | — |
| BFN（best-of-N，原始动作空间）| 51% | 63% | — |
| FQL-n（biased n-step）| 27% | 57% | n-step 有偏，不稳定 |
| **QC-FQL**（Q-chunking）| 38% | **86%** | +28pp vs FQL |
| **QC**（Q-chunking）| 52% | **86%** | +23pp vs BFN |

最难任务 cube-quadruple：QC 73%，QC-FQL 77%，BFN 仅 12%。

### 关键消融

- chunk 长度 $h$：$h=10$ 峰值最优；$h=50$ 完全失败（策略反应性损失过大）
- Gaussian policy 在 chunk RL 中效果差（Q-chunking 必须用 flow/diffusion 策略捕获复杂分布）
- n-step return 相比 Q-chunking 更不稳定（off-policy bias 显著）

---

## 与相关概念的关系

- **[[wiki/concepts/imitation-learning/ACT]]**：Action Chunking 在模仿学习中的来源，Q-chunking 将其引入 TD-based RL
- **[[wiki/concepts/rl/DPPO]]**：PPO 微调扩散策略的对比方向，Q-chunking 聚焦离线→在线 Q-learning 而非 on-policy PG
- **[[wiki/concepts/benchmarks/OGBench]]**：Q-chunking 的主要评测基准（来源 47/48/49 均在此评测）
- **[[wiki/concepts/generative-models/Flow Matching]]**：Q-chunking 行为策略的参数化选择（Gaussian 不够表达）
