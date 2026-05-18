---
type: analysis
tags: [RL-Finetuning, Flow Matching, Diffusion Policy, BPTT, Policy Gradient, Offline RL, On-Policy, 生成模型, 策略优化]
sources:
  - raw/sources/papers/VLA+RL/Ren 等 - 2024 - Diffusion Policy Policy Optimization.pdf
  - raw/sources/papers/VLA+RL/Zhang 等 - 2026 - ReinFlow Fine-tuning Flow Matching Policy with Online Reinforcement Learning.pdf
  - raw/sources/papers/VLA+RL/Park 等 - 2025 - Flow Q-Learning.pdf
  - raw/assets/papers/VLA+RL/Gao - 2026 - FlowRL.pdf
  - raw/assets/papers/VLA+RL/McAllister - 2025 - FPO.pdf
  - raw/assets/papers/VLA+RL/Kang - 2026 - WarmPrior.pdf
  - raw/assets/papers/VLA+RL/Nguyen - 2025 - OFQL.pdf
  - raw/assets/papers/VLA+RL/Lee - 2026 - FAN.pdf
created: 2026-05-18
updated: 2026-05-18
---

# RL 微调生成模型：技术挑战与解法全景

> **问题焦点**：以扩散模型（Diffusion Policy）和流匹配模型（Flow Matching）为代表的生成策略，在机器人学习中展现出卓越的多模态动作表达能力。然而，用强化学习对其进行在线/离线微调，面临一系列深层次技术障碍——本文系统梳理这些障碍的根本原因，以及各主流方法（ReinFlow、FQL、DPPO、FPO、OFQL、FAN、WarmPrior）如何从不同角度加以破解。

---

## 一、为什么要 RL 微调生成模型

扩散策略与 Flow Matching 策略通过 **行为克隆（BC）** 预训练，能够精确刻画演示数据的多模态动作分布。但 BC 有根本性局限：

- **分布外泛化能力差**：策略只会模仿训练分布，遇到没见过的状态会失效（Compound Error）
- **无法超越演示**：奖励信号是任务完成的直接反馈，而 BC 只能达到演示者水平

RL 微调提供了**突破演示质量上限**的可能，但对生成策略的 RL 微调并非直接套用标准 RL 算法即可。

---

## 二、技术障碍：RL 微调生成策略的核心难题

### 难题 1：边缘 Log 概率不可解析

**根本原因**：生成策略通过**多步迭代过程**生成动作：

$$a^{(0)} \sim \mathcal{N}(0, I) \xrightarrow{\;f_\theta\;} a^{(1)} \xrightarrow{\;f_\theta\;} \cdots \xrightarrow{\;f_\theta\;} a^{(T)} = a$$

最终输出动作 $a$ 的 log 概率 $\log \pi_\theta(a \mid s)$ 需对所有中间轨迹积分：

$$\log \pi_\theta(a \mid s) = \log \int \prod_{k=1}^{T} p_\theta(a^{(k)} \mid a^{(k-1)}, s)\,\mathrm{d}a^{(1:T-1)}$$

这个积分**没有解析封闭形式**。而标准策略梯度（如 PPO、TRPO）的目标函数恰恰依赖 $\log \pi_\theta(a \mid s)$ 来计算重要性采样比：

$$\rho_\theta = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}$$

对于高斯策略，$\log \pi_\theta$ 一行就能写出；对于生成策略，这是整个难题的核心。

### 难题 2：BPTT 的三重代价

**BPTT（Backpropagation Through Time）** 是处理时间展开迭代模型的反向求导算法。对于 $T$ 步生成过程，损失 $\mathcal{L}$ 对参数 $\theta$ 的梯度必须穿透整条迭代链：

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial a^{(T)}} \cdot \underbrace{\prod_{k=t+1}^{T} \frac{\partial a^{(k)}}{\partial a^{(k-1)}}}_{\text{穿透 } T-t \text{ 步}} \cdot \frac{\partial a^{(t)}}{\partial \theta}$$

这在生成策略中引入三重代价：

| 代价维度     | 具体表现                                                                                     | 量级                         |
| -------- | ---------------------------------------------------------------------------------------- | -------------------------- |
| **显存爆炸** | 必须保存所有 $T$ 步中间激活以备反向传播                                                                   | 随 $T$ **线性增长**，VLA 大模型尤为严重 |
| **梯度病态** | 雅可比矩阵 $\prod_k \frac{\partial a^{(k)}}{\partial a^{(k-1)}}$ 的特征值若偏离 1，梯度在 $T$ 步后指数级消失或爆炸 | 随 $T$ **指数放大**             |
| **计算时间** | 每个训练 step 需完整执行 $T$ 次前向 + $T$ 次反向                                                        | 训练比 BC **慢 $T$ 倍**         |

> **典型数字**：Flow Matching 策略的推理步数 $T = 4 \sim 50$，DDPM 则高达 $T = 100$。BPTT 的代价在这个规模下是完全不可接受的。

### 难题 3：探索-利用悖论

**Diffusion Policy** 通过随机去噪过程自带一定随机性，但其噪声 schedule 在 BC 预训练后被固化，**RL 微调的探索可能破坏流形结构**，导致策略偏离数据支撑集。

**Flow Matching** 更为严峻：Flow 策略使用**确定性 ODE**，给定初始噪声和观测，输出动作是完全确定的：

$$\frac{da}{dt} = v_\theta(t, a, s), \quad a(0) \sim \mathcal{N}(0, I)$$

这意味着 Flow 策略**没有任何内置的探索机制**，与 RL 的探索-利用平衡根本矛盾。

### 难题 4：离线 Q-Learning 的策略抽取问题

在 **离线 RL** 框架中，即使已经学到了 Q 函数 $Q(s, a)$，从中抽取最优策略也面临挑战：

$$\pi^* = \arg\max_\pi\, \mathbb{E}_{a \sim \pi}[Q(s, a)]$$

对于高斯策略，$\arg\max$ 通过简单的重参数化梯度就能实现。但对于生成策略：

- 动作 $a$ 由 $T$ 步 ODE 生成，$\nabla_\theta Q(s, a(\theta))$ 需要 BPTT 穿透 $T$ 步
- 直接对 Flow/Diffusion 策略做 Q-value 最大化 = **重新陷入 BPTT 困境**

---

## 三、解法全景：三条技术路线

面对上述四大难题，当前主流方法形成了三条**设计哲学截然不同**的技术路线。

```
  在线 RL（有模拟器）                           离线 RL（静态数据集）
  ─────────────────────────────────────────────────────────
  路线 A：Markov 化去噪链          路线 C：完全回避 BPTT → 离线 Q-Learning
    DPPO ──── ReinFlow                  FQL ──── OFQL ──── FAN
  ─────────────────────────────────────────────────────────
  路线 B：CFM loss ratio 近似 IS ratio
    FPO（采样器无关）
  ─────────────────────────────────────────────────────────
  正交优化轴：WarmPrior（源分布替换，可叠加于任意方法）
```

---

### 路线 A：Markov 化去噪链（显式 log-prob）

**哲学**：与其绕开 log-prob 问题，不如将迭代生成过程**重新建模**为 Markov 过程，让每一步都具有可解析的高斯似然，从而启用标准策略梯度。

#### A1. DPPO：两层 MDP

[[wiki/sources/rl-finetuning/2026-04-19 DPPO]] 的核心洞察：DDPM 的去噪过程**本身就是一个 Markov 链**——每一步 $p_\theta(a^{k-1} \mid a^k, o)$ 已经是高斯分布，可以直接写出 log-prob。

**两层 MDP 结构**：

$$\text{外层 MDP（环境）} \supset \text{内层 MDP（去噪）}$$

$$\log \pi_\theta(\mathcal{A} \mid o) = \sum_{k=1}^{K} \log \mathcal{N}\!\left(a^{k-1};\; \mu_\theta(a^k, o, k),\; \sigma_k^2 I\right)$$

**PPO 更新目标**：

$$\max_\theta\; \mathbb{E}\!\left[\min\!\left(\rho_\theta \hat{A},\; \mathrm{clip}(\rho_\theta, 1-\varepsilon, 1+\varepsilon)\hat{A}\right)\right], \quad \rho_\theta = \prod_{k=1}^{K} \frac{\pi_\theta(a^{k-1} \mid a^k, o)}{\pi_{\theta_\mathrm{old}}(a^{k-1} \mid a^k, o)}$$

**解决了哪些难题**：

| 难题 | DPPO 的解法 |
|---|---|
| Log-prob 不可解析 | DDPM 每步自带高斯，绑定 DDPM schedule 获得可解析似然 |
| BPTT | 两层 MDP 展开后，策略梯度只在**单步**高斯内传播，无需穿透整条链 |
| 探索 | 修改 noise schedule 在微调阶段注入适度随机性；扩散策略沿数据流形的**结构化探索**优于高斯策略 |

**代价**：深度绑定 DDPM 的噪声 schedule，不能直接用于 Flow Matching 策略。

#### A2. ReinFlow：噪声注入 Markov 化

[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] 把 DPPO 的思路迁移到 Flow Matching——但 Flow 的 ODE 是确定性的，没有天然的随机性。解法：**主动注入可学习噪声**，将确定性 ODE 转化为随机过程。

**带噪声的 Flow 步**：

$$a_{k+1} = \underbrace{a_k + v_\theta(t_k, a_k, o)\,\Delta t_k}_{\text{ODE 步（原 Flow 策略）}} + \underbrace{\sigma_{\theta'}(t_k, a_k, o)\,\varepsilon}_{\text{可学习噪声}}, \quad \varepsilon \sim \mathcal{N}(0, I)$$

其中 $v_\theta$ 是预训练 Flow 策略（固定），$\sigma_{\theta'}$ 是**只在 RL 微调阶段引入的噪声网络**。

注入噪声后，整条去噪路径变为 Markov 过程，log-prob 具有封闭形式：

$$\ln \pi(a_0,\ldots,a_K \mid o) = \ln \mathcal{N}(a_0; 0, I) + \sum_{k=0}^{K-1} \ln \mathcal{N}\!\left(a_{k+1};\; a_k + v_\theta\,\Delta t_k,\; \sigma_{\theta'}^2\right)$$

**解决了哪些难题**：

| 难题 | ReinFlow 的解法 |
|---|---|
| Log-prob 不可解析 | 噪声注入后变为离散 Markov 过程，封闭形式 log-prob 可计算 |
| BPTT | 策略梯度仅在单步噪声内传播（Markov Property PG 定理 4.1） |
| 探索 | $\sigma_{\theta'}$ 提供**可调节的学习式探索**，噪声幅度是关键超参数 |
| Flow 无 log-prob | 通过噪声注入从根本上解决，无需修改 ODE 积分器 |

**额外优势**：$\sigma_{\theta'}$ 在 RL 微调完成后**可以直接丢弃**，推理时完全退回到原始 ODE，不引入任何额外延迟。支持最少 **1 步**去噪，比 DPPO 快 **62.82%**（全任务平均）。

---

### 路线 B：CFM Loss Ratio 近似 IS Ratio（FPO）

**哲学**：不通过 Markov 化计算 log-prob，而是观察到**条件流匹配训练损失本身与重要性采样比在结构上等价**，直接用 loss 比值替代 IS ratio。

[[wiki/sources/rl-finetuning/2026-05-17 FPO Flow Matching Policy Gradients (McAllister 2025)]] 的核心洞察：

标准 PPO 使用的重要性采样比为 $\rho_\theta = \frac{\pi_\theta(a)}{\pi_{\theta_\text{old}}(a)}$。

条件流匹配（CFM）的训练目标是速度场的 MSE：

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_t}\!\left[\|v_\theta(t, x_t) - u(t, x_t \mid x_1)\|^2\right]$$

FPO 以 CFM loss 的比值构造 surrogate：

$$r_t(\theta) = \frac{\mathcal{L}_{\text{CFM}}(a_t, s_t;\; \theta_{\text{old}})}{\mathcal{L}_{\text{CFM}}(a_t, s_t;\; \theta)}, \qquad \mathcal{L}_{\text{FPO}} = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

这形式上与 PPO-clip **完全相同**，但完全不依赖 log-prob。

**解决了哪些难题**：

| 难题 | FPO 的解法 |
|---|---|
| Log-prob 不可解析 | 完全绕开，不计算 log-prob |
| 采样器绑定 | ✅ 采样器无关（agnostic）：训练完成后可自由切换任意 ODE 积分器 |
| BPTT | 用 CFM loss 作为优化代理，无需穿透 ODE |

**代价**：CFM loss ratio 近似 IS ratio 存在**理论误差**，理论保证弱于 A 路线的精确 log-prob 方案。

---

### 路线 C：完全回避 BPTT → 离线 Q-Learning 族

**哲学**：既然 BPTT 代价高昂，那就换一条完全不需要策略梯度的路——用 Q-learning 驱动策略改进，同时通过巧妙的架构设计把表达性生成模型与 Q 值优化**完全解耦**。

路线 C 分为三个子方案，代表离线 Q-learning 族的演进方向：

#### C1. FQL：BC-Flow 与单步策略解耦

[[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]] 的关键洞察：**把"表达性"和"可微性"交给两个不同的网络**。

- **BC Flow 策略 $\mu^\beta(s, z)$**：用行为克隆训练，只负责刻画数据分布的多模态性，不参与 Q 值优化
- **单步策略 $\mu(s, z)$**：可微，只负责最大化 Q 值，同时蒸馏自 BC Flow 以保持数据约束

$$\min_\mu\; \underbrace{\mathbb{E}[-Q(s, \mu(s, z))]}_{\text{Q 最大化（单步可微，无 BPTT）}} + \alpha\,\underbrace{\mathbb{E}[\|\mu(s, z) - \mu^\beta(s, z)\|^2]}_{\text{蒸馏正则（保持行为约束）}}$$

因为 $\mu(s, z)$ 是单步网络，$\nabla_\theta Q(s, \mu_\theta(s, z))$ 只穿透**一次**前向传播，完全消除 BPTT。

**解决了哪些难题**：

| 难题 | FQL 的解法 |
|---|---|
| BPTT | 单步策略完全消除 BPTT，Q 梯度只穿 1 层 |
| Log-prob 不可解析 | 不需要——离线 Q-learning 不用 log-prob |
| 推理延迟 | 部署时直接用 $\mu(s, z)$，**单步推理**，无需 ODE 积分 |
| 多模态表达 | BC Flow 保留，通过蒸馏传递给单步策略 |

#### C2. OFQL：Marginal Average Velocity Field

[[wiki/sources/rl-finetuning/2026-05-17 OFQL (Nguyen 2025)]] 发现 FQL 的两阶段流程（先预训 BC Flow，再蒸馏）增加了工程复杂度。更激进的方案：**直接学习一个可以单步映射的速度场**。

标准流匹配学习**条件速度场** $u(x_t \mid x_0, x_1)$（路径依赖，单步近似误差大）。OFQL 改为学习**边缘平均速度场**：

$$\bar{u}_\theta(x_t) = \mathbb{E}_{x_0, x_1}[u(x_t \mid x_0, x_1) \mid x_t]$$

训练目标：

$$\mathcal{L}_{\text{OFQL}} = \mathbb{E}_{t, x_0, x_1}\!\left[\|\bar{u}_\theta(x_t) - (x_1 - x_0)\|^2\right]$$

推理时只需一次前向：

$$a = x_0 + \bar{u}_\theta(x_0), \quad x_0 \sim \mathcal{N}(0, I)$$

这是从理论上保证的"直接映射"，无需迭代 ODE，无需蒸馏，无需两阶段训练。

#### C3. FAN：噪声条件化分布式 Critic

[[wiki/sources/rl-finetuning/2026-05-17 FAN (Lee 2026)]] 在 OFQL 的单步流基础上，引入了对 **分布式 Critic** 的系统性简化。

分布式 critic（如 IQN）通常需要多个分位数样本 $\{\tau_i\}$ 估计 Q 分布，计算代价高。FAN 的洞察：将**噪声本身作为 critic 输入**，单个噪声样本即可参数化完整的 Q 分布：

$$Q_\phi(s, a, \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

完整目标函数：

$$\mathcal{L}_{\text{FAN}} = -\mathbb{E}[Q_\phi(s, a, \epsilon)] + \lambda\,\mathcal{L}_{\text{BC}}$$

FAN 提供了正式的**收敛性证明与性能界**，是三个离线方法中理论保证最强的。

**路线 C 三者演进对比**：

| 维度 | FQL | OFQL | FAN |
|---|---|---|---|
| 单步实现 | BC Flow 预训练 + 蒸馏（两阶段） | Average velocity field（端到端） | Flow-anchored（端到端） |
| Critic 类型 | 标准 Q | 标准 Q | Noise-conditioned $Q(s, a, \epsilon)$ |
| 工程复杂度 | 较高（两阶段） | 中等 | 中等 |
| 理论保证 | 蒸馏近似，较弱 | 未报告 | ✅ 收敛性 + 性能界 |

---

### 正交优化轴：WarmPrior（源分布设计）

[[wiki/sources/rl-finetuning/2026-05-17 WarmPrior (Kang 2026)]] 提出了一个与上述三条路线**完全正交**的改进轴：优化流匹配的**源分布设计**。

**问题根源**：标准流匹配使用 $\mathcal{N}(0, I)$ 作为源分布，而机器人动作在时序上高度相关——相邻时刻的动作极为相近，高斯噪声对此一无所知。这导致 ODE 路径弯曲，需要更多积分步数，也增大了单步近似误差。

**WarmPrior 解法**：用近期动作历史构造**时序先验**替换标准高斯：

$$q_{\text{WarmPrior}} = \mathcal{N}(\mu_{\text{hist}},\; \sigma^2 I)$$

其中 $\mu_{\text{hist}}$ 从近期 $k$ 步动作历史计算（均值或最后一步）。

| 机制 | 效果 |
|---|---|
| 源与目标动作分布更接近 | ODE 路径更短更直（Rectified Flow 理论保证）|
| 历史动作包含时序连续性先验 | 流学习难度降低，BC 性能提升 |
| 路径更直 → 单步近似误差更小 | 推理步数可减少，或单步精度更高 |
| RL 中作为探索先验 | 探索集中在高质量动作附近，样本效率提升 |

**可叠加性**：WarmPrior 不修改优化目标，只改变采样起点，与路线 A/B/C 的任意方法完全兼容，是一个"免费"的正交改进。

---

## 四、三条路线的根本分歧

从设计哲学层面，三条路线对"RL 微调生成模型"的核心问题有截然不同的判断：

| 判断 / 哲学 | 路线 A（Markov 化） | 路线 B（CFM loss ratio） | 路线 C（离线 Q-learning） |
|---|---|---|---|
| **如何处理 log-prob** | 将生成过程 Markov 化以获得解析 log-prob | 用 CFM loss 比值近似 IS ratio，完全绕开 | 离线 Q-learning 不需要 log-prob |
| **如何处理 BPTT** | Markov 展开后梯度仅在单步传播，无全链 BPTT | 同上 | 单步策略 / average velocity field 从架构上消除 |
| **如何探索** | 主动注入可学习噪声（ReinFlow）或利用扩散内置噪声（DPPO） | 利用流匹配采样随机性 | 不做在线探索；探索依赖离线数据集质量 |
| **代价** | 绑定特定 schedule；需额外噪声网络（ReinFlow） | IS ratio 近似误差；理论保证较弱 | 无法超越离线数据集分布；FQL 需两阶段训练 |
| **适用场景** | 有仿真器，在线探索可行 | 有仿真器，需要采样器无关性 | 只有静态数据集，需要快速推理 |

---

## 五、严谨数学推导

### 5.1 符号定义

| 符号 | 含义 |
|---|---|
| $s$ | 状态（观测） |
| $a$ | 最终输出动作 |
| $a^{(k)}$ | 第 $k$ 步中间去噪状态，$a^{(0)} \sim \mathcal{N}(0, I)$，$a^{(T)} = a$ |
| $\theta$ | 策略参数 |
| $v_\theta(t, x, s)$ | Flow Matching 的速度场网络 |
| $\sigma_{\theta'}(t, x, s)$ | ReinFlow 的可学习噪声网络 |
| $Q(s, a)$ | 状态-动作价值函数 |
| $\hat{A}$ | Advantage 估计（GAE） |
| $\pi_\theta(a \mid s)$ | 策略（最终动作的边缘分布） |
| $T$ | 去噪/积分步数 |

### 5.2 为什么生成策略的 log-prob 不可解析：完整推导

**目标**：计算 $\log \pi_\theta(a \mid s)$，其中 $a = a^{(T)}$ 由迭代过程产生。

**推导**：

$$\pi_\theta(a \mid s) = \int p(a^{(0)}) \prod_{k=0}^{T-1} p_\theta(a^{(k+1)} \mid a^{(k)}, s)\,\mathrm{d}a^{(0:T-1)}$$

对于确定性 ODE（Flow Matching），每步转移退化为 Dirac delta：

$$p_\theta(a^{(k+1)} \mid a^{(k)}, s) = \delta\!\left(a^{(k+1)} - a^{(k)} - v_\theta(t_k, a^{(k)}, s)\Delta t\right)$$

取 log 后，积分消失但 $a$ 与 $a^{(0)}$ 之间存在复杂的非线性映射。对于 $T$ 步 ODE，从 $a^{(0)}$ 到 $a = a^{(T)}$ 的映射 $\Phi_\theta: a^{(0)} \mapsto a$ 是 $T$ 次迭代的复合函数：

$$\log \pi_\theta(a \mid s) = \log p(a^{(0)}) - \log \left|\det \frac{\partial \Phi_\theta}{\partial a^{(0)}}\right|$$

其中 $a^{(0)} = \Phi_\theta^{-1}(a)$，而 $\left|\det \frac{\partial \Phi_\theta}{\partial a^{(0)}}\right|$ 是 $T$ 次雅可比矩阵之积的行列式，$T$ 较大时**数值计算极其昂贵**（$\mathcal{O}(d^3 T)$，$d$ 为动作维度）。

**结论**：对于确定性 ODE，精确计算 $\log \pi_\theta(a \mid s)$ 的代价与 $d^3 T$ 成正比，是完全不可行的。

### 5.3 DPPO 的两层 MDP 使 log-prob 可解析：推导

DDPM 的去噪步骤为：

$$q(a^{k-1} \mid a^k) = \mathcal{N}\!\left(a^{k-1};\; \mu_\theta(a^k, s, k),\; \sigma_k^2 I\right)$$

因此**每步的 log-prob 有封闭形式**：

$$\log \pi_\theta(a^{k-1} \mid a^k, s) = -\frac{d}{2}\log(2\pi\sigma_k^2) - \frac{\|a^{k-1} - \mu_\theta(a^k, s, k)\|^2}{2\sigma_k^2}$$

整条去噪路径的联合 log-prob（Markov 性）：

$$\log \pi_\theta(a^{0:K} \mid s) = \sum_{k=0}^{K-1} \log \pi_\theta(a^{k-1} \mid a^k, s)$$

PPO 的重要性采样比：

$$\rho_\theta = \frac{\pi_\theta(a^{0:K} \mid s)}{\pi_{\theta_\text{old}}(a^{0:K} \mid s)} = \exp\!\left(\sum_{k=0}^{K-1}\left[\log \pi_\theta(a^{k-1} \mid a^k, s) - \log \pi_{\theta_\text{old}}(a^{k-1} \mid a^k, s)\right]\right)$$

这是完全可解析的，**每步梯度仅穿透单个高斯**，不发生 BPTT 中的多步链式雅可比乘积。

### 5.4 ReinFlow 噪声注入的 Markov 化推导

噪声注入后的随机过程：

$$a_{k+1} = a_k + v_\theta(t_k, a_k, s)\Delta t_k + \sigma_{\theta'}(t_k, a_k, s)\,\varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, I)$$

对固定的 $a_k$，$a_{k+1}$ 服从：

$$p_{\theta, \theta'}(a_{k+1} \mid a_k, s) = \mathcal{N}\!\left(a_{k+1};\; a_k + v_\theta \Delta t_k,\; \sigma_{\theta'}^2 I\right)$$

这是一个高斯分布，log-prob 有封闭形式：

$$\log p(a_{k+1} \mid a_k, s) = -\frac{d}{2}\log(2\pi\sigma_{\theta'}^2) - \frac{\|a_{k+1} - a_k - v_\theta \Delta t_k\|^2}{2\sigma_{\theta'}^2}$$

整条路径的联合 log-prob（Markov 性保证可分解）：

$$\log \pi_{rl}(a_0, \ldots, a_K \mid s) = \log \mathcal{N}(a_0; 0, I) + \sum_{k=0}^{K-1} \log \mathcal{N}\!\left(a_{k+1};\; a_k + v_\theta \Delta t_k,\; \sigma_{\theta'}^2 I\right)$$

**关键结论**（Markov Process PG 定理 4.1）：对于上述由离散时间 Markov 过程参数化的策略，策略梯度等价于：

$$\nabla_{\theta, \theta'} J = \mathbb{E}\!\left[\nabla_{\theta, \theta'} \log \pi_{rl}(a_0, \ldots, a_K \mid s) \cdot \hat{A}\right]$$

其中 $\hat{A}$ 仅基于最终动作的环境奖励计算，不依赖中间步的奖励信号。这个梯度只在**单步内的高斯参数**上传播，完全避免了 BPTT。

### 5.5 FQL 解耦如何消除 BPTT

FQL 的优化目标：

$$\min_\mu\; \mathbb{E}_{z \sim \mathcal{N}}\!\left[-Q(s, \mu(s, z)) + \alpha\|\mu(s, z) - \mu^\beta(s, z)\|^2\right]$$

梯度：

$$\nabla_\theta \mathcal{L}_{\text{FQL}} = -\nabla_a Q(s, a)\big|_{a=\mu(s,z)} \cdot \nabla_\theta \mu(s, z) + \alpha \cdot 2(\mu_\theta(s, z) - \mu^\beta(s, z)) \cdot \nabla_\theta \mu_\theta(s, z)$$

关键：$\mu(s, z)$ 是**单步网络**（一次前向），$\nabla_\theta \mu(s, z)$ 只穿透单层网络，不涉及任何迭代过程。BC Flow $\mu^\beta$ 视为目标（停梯度），蒸馏项同样不触发 BPTT。

**与直接 BPTT 的对比**：若直接对多步 Flow $\pi_\theta$ 做 $-Q(s, a)$ 梯度：

$$\nabla_\theta [-Q(s, a^{(T)})] = -\nabla_a Q \cdot \underbrace{\frac{\partial a^{(T)}}{\partial \theta}}_{\text{穿透 } T \text{ 步 ODE，梯度链 } T \text{ 层深}}$$

FQL 将这个代价从 $\mathcal{O}(T)$ 降低到 $\mathcal{O}(1)$。

---

## 六、总结与前瞻

### 三条路线的实用选型

| 场景 | 推荐方法 | 核心理由 |
|---|---|---|
| 有仿真器，DDPM 基线 | **DPPO** | 成熟事实基线，结构化探索优势 |
| 有仿真器，Flow 策略，追求推理速度 | **ReinFlow** | 1 步推理，wall-clock -62%，理论完备 |
| 有仿真器，Flow 策略，需采样器灵活性 | **FPO** | 训练后可自由切换积分器 |
| 静态数据集，多模态分布，需 offline→online | **FQL** | 蒸馏解耦，offline-to-online 支持 |
| 静态数据集，需单步推理，无两阶段训练 | **OFQL / FAN** | 端到端单步，无蒸馏开销 |
| 需要最强理论保证 | **FAN** | 收敛性 + 性能界均有证明 |
| 任何 Flow 方法的正交加速 | **WarmPrior** | 叠加式改进，无架构修改 |

### 仍然悬而未决的问题

1. **在线 vs. 离线的质量上限**：路线 C 的离线方法受限于数据集覆盖度，路线 A 的在线方法则依赖高质量仿真器——真实机器人场景如何兼得两者的优势？
2. **大模型规模的 BPTT 瓶颈**：当策略骨干是 7B+ 的 VLA 时，即使是单步内的梯度传播也会产生严峻的显存压力——VLA 的 RL 微调（RECAP、πRL 等）选择了完全不同的路线（advantage conditioning），这是否意味着 PG 方法在 VLA 规模上根本不可行？
3. **WarmPrior 的泛化边界**：时序先验假设机器人动作高度连续，对于需要**急停/方向突变**的精细操作任务，WarmPrior 是否会引入偏差？

---

## 关联页面

- **方法综合对比**：[[wiki/comparisons/RL 微调表达性策略方法对比]]
- **VLA RL 微调（另一赛道）**：[[wiki/comparisons/VLA RL 微调方法对比]]
- **策略参数化基础**：[[wiki/concepts/generative-models/Flow Matching]]、[[wiki/concepts/generative-models/Diffusion Policy]]
- **RL 范式**：[[wiki/concepts/rl/Offline 强化学习]]、[[wiki/concepts/rl/DPPO]]
- **数学推导深化**：[[wiki/analyses/DPPO 完整数学推导]]、[[wiki/analyses/ReinFlow π_rl 完整数学推导]]
- **来源页面**：[[wiki/sources/rl-finetuning/2026-04-19 DPPO]]、[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]]、[[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]]、[[wiki/sources/rl-finetuning/2026-05-17 FPO Flow Matching Policy Gradients (McAllister 2025)]]、[[wiki/sources/rl-finetuning/2026-05-17 WarmPrior (Kang 2026)]]、[[wiki/sources/rl-finetuning/2026-05-17 OFQL (Nguyen 2025)]]、[[wiki/sources/rl-finetuning/2026-05-17 FAN (Lee 2026)]]
