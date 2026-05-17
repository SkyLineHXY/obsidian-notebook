---
type: analysis
tags:
  - DPPO
  - PPO
  - 扩散策略
  - 策略梯度
  - 强化学习
  - 数学推导
  - 双层MDP
  - 重要性采样
sources:
  - raw/sources/papers/VLA+RL/Ren 等 - 2024 - Diffusion Policy Policy Optimization.pdf
  - raw/sources/papers/Generative Model/Ho 等 - 2020 - Denoising Diffusion Probabilistic Models.pdf
created: 2026-04-23
updated: 2026-04-23
---

# DPPO 完整数学推导

> **定位**：本页面是 [[wiki/concepts/DPPO]] 的零基础完整数学展开。目标是让你不仅会用公式，还能理解每一步从哪里来、为什么这样做。
> 关联页面：[[wiki/concepts/DPPO]] · [[wiki/concepts/DDPM]] · [[wiki/concepts/Diffusion Policy]] · [[wiki/analyses/Flow Matching 完整数学推导]]

---

## 0. 符号定义

在阅读推导前，先把所有符号的含义固定下来：

| 符号                                          | 含义                                                                |
| ------------------------------------------- | ----------------------------------------------------------------- |
| $s_t \in \mathcal{S}$                       | 环境时刻 $t$ 的状态（机器人观测）                                               |
| $a_t \in \mathcal{A}$                       | 时刻 $t$ 执行的动作（最终去噪结果，即 $a_t = a^0$）                                |
| $r_t \in \mathbb{R}$                        | 执行 $a_t$ 后环境返回的即时奖励                                               |
| $\gamma \in (0,1)$                          | 折扣因子（未来奖励的衰减率）                                                    |
| $\pi_\theta$                                | 参数为 $\theta$ 的策略（即扩散网络）                                           |
| $V_\phi(s_t)$                               | 参数为 $\phi$ 的价值网络，估计从 $s_t$ 出发的期望折扣回报                              |
| $K$                                         | 扩散去噪总步数（DDPM 取 100，DDIM 取 10～20）                                  |
| $k$                                         | 当前去噪步索引，$k = K \to K{-}1 \to \cdots \to 1 \to 0$（逆时间方向）           |
| $a^k$                                       | 第 $k$ 步去噪中间状态（$a^K \sim \mathcal{N}(0,I)$ 是纯噪声，$a^0$ 是最终动作）       |
| $\mu_\theta(a^k, s, k)$                     | 神经网络预测的第 $k$ 步去噪均值                                                |
| $\sigma_k$                                  | 第 $k$ 步去噪的固定标准差（由噪声调度决定）                                          |
| $\beta_k \in (0,1)$                         | 前向加噪方差调度（固定超参数）                                                   |
| $\alpha_k = 1-\beta_k$                      | 每步信号保留比例                                                          |
| $\bar{\alpha}_k = \prod_{i=1}^{k} \alpha_i$ | 累积信号保留比例                                                          |
| $\epsilon_\theta(a^k, s, k)$                | 噪声预测神经网络（U-Net 或 Transformer）                                     |
| $d$                                         | 动作向量的维度                                                           |
| $\rho_\theta^k$                             | 第 $k$ 步去噪的新旧策略重要性采样比                                              |
| $\hat{A}_t$                                 | 状态 $s_t$ 下的优势函数估计值                                                |
| $\epsilon_{clip} \in (0,1)$                 | PPO 裁剪范围超参数（通常取 $0.2$；为避免与噪声 $\epsilon$ 混淆，本页用 $\epsilon_{clip}$） |

---

## 1. 预备知识 A：强化学习与策略梯度

### 1.1 强化学习的目标

在强化学习（RL）中，智能体通过与环境反复交互，学习最大化长期累积奖励。形式上，这对应一个**马尔可夫决策过程（MDP）**：

- 智能体在状态 $s_t$ 下，按策略 $\pi_\theta$ 采样动作 $a_t \sim \pi_\theta(\cdot | s_t)$
- 环境执行动作后，转移到新状态 $s_{t+1}$，同时给出奖励 $r_t$
- 目标：最大化期望折扣累积回报

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

其中 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ 是交互轨迹，$\gamma^t$ 对未来奖励打折扣（离当下越远，越不确定）。

---

### 1.2 策略梯度定理（Policy Gradient Theorem）

**问题**：如何计算 $\nabla_\theta J(\theta)$？

**答案**（Williams 1992；Sutton 等 1999）：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right]$$

其中 **Q 函数** $Q^{\pi_\theta}(s_t, a_t) = \mathbb{E}\!\left[\sum_{t'=t}^T \gamma^{t'-t} r_{t'}\right]$ 是"从 $s_t$ 执行 $a_t$ 后的期望回报"。

**直觉**：$\nabla_\theta \log \pi_\theta(a_t | s_t)$ 是增大动作 $a_t$ 概率的方向；乘以 $Q$ 之后，好结果的动作被放大，差结果的动作被缩小。

**优势函数（Advantage）**：为减少方差，令：

$$A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)$$

其中 **价值函数** $V^{\pi_\theta}(s_t) = \mathbb{E}_{a \sim \pi_\theta}[Q^{\pi_\theta}(s_t, a)]$ 是状态 $s_t$ 的"平均回报基线"。$A$ 衡量"这个动作比平均水平**好多少**"。

用 $\hat{A}_t$（优势函数估计）替代 $Q$，梯度变为：

$$\nabla_\theta J(\theta) \approx \mathbb{E}_{(s_t,a_t) \sim \pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t\right]$$

这就是 **REINFORCE** 算法的核心更新公式。

---

### 1.3 重要性采样（Importance Sampling）

**问题**：每次更新 $\theta$ 后，旧轨迹数据就"过期"了，需要重新采集数据——**样本效率极低**。

**解决思路**：能否用**旧策略 $\pi_{\theta_{old}}$** 采集的数据，来估计**新策略 $\pi_\theta$** 的梯度？

**重要性采样恒等式**：对于任意函数 $f(a)$，

$$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_{old}}}\!\left[\underbrace{\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}}_{\rho_\theta(s,a)} \cdot f(a)\right]$$

**推导**（展开期望定义）：

$$\int \pi_\theta(a|s)\, f(a)\, da = \int \pi_{\theta_{old}}(a|s) \cdot \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \cdot f(a)\, da = \mathbb{E}_{a \sim \pi_{\theta_{old}}}\!\left[\rho_\theta \cdot f(a)\right]$$

于是策略目标化简为：

$$L^{IS}(\theta) = \mathbb{E}_{(s_t,a_t) \sim \pi_{\theta_{old}}}\!\left[\rho_\theta(s_t, a_t) \cdot \hat{A}_t\right], \quad \rho_\theta = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

这允许我们**复用旧数据多次更新**，大大提升样本效率。

---

### 1.4 PPO（近端策略优化）

**问题**：当 $\rho_\theta$ 偏离 1 太多时，重要性采样误差变大，更新不稳定（旧数据与新策略分布差异太大）。

**PPO-Clip 的解法**：用硬性裁剪（clip）限制 $\rho_\theta$ 的范围，只接受"小步"更新：

$$\boxed{L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(\rho_\theta\, \hat{A}_t,\;\; \mathrm{clip}(\rho_\theta,\, 1{-}\epsilon_{clip},\, 1{+}\epsilon_{clip})\, \hat{A}_t\right)\right]}$$

**逐项理解**：

| 情况                            | $\hat{A}_t > 0$（好动作） | $\hat{A}_t < 0$（坏动作） |
| ----------------------------- | -------------------- | -------------------- |
| $\rho_\theta$ 较小（新策略不太可能采此动作） | 有提升空间，正常更新           | 避免过度惩罚，clip 截断       |
| $\rho_\theta$ 较大（新策略大幅偏向此动作）  | 避免过度强化，clip 截断       | 有空间进一步压制             |

取 $\min$ 保证目标函数是一个保守的下界：**只接受"有益且安全"的更新**。

**为什么 PPO 比 REINFORCE 好**：PPO 允许用一批数据做多次迭代，每次保持在旧策略附近，既高效又稳定。

---

## 2. 预备知识 B：扩散策略的数学结构

### 2.1 前向加噪过程（复习 DDPM）

DDPM 的**前向过程**将干净动作 $a^0$ 逐步加噪：

$$q(a^k | a^{k-1}) = \mathcal{N}\!\left(a^k;\; \sqrt{1-\beta_k}\, a^{k-1},\; \beta_k I\right)$$

利用高斯函数的可加性，可以跳过中间步骤，直接从 $a^0$ 采样任意时刻 $a^k$：

$$a^k = \sqrt{\bar{\alpha}_k}\, a^0 + \sqrt{1-\bar{\alpha}_k}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \tag{前向闭式}$$

其中 $\bar{\alpha}_k = \prod_{i=1}^k (1-\beta_i)$。当 $k$ 很大时，$\bar{\alpha}_k \to 0$，$a^k \to$ 纯高斯噪声。

---

### 2.2 逆向去噪过程（Diffusion Policy 的动作生成）

**Diffusion Policy** 将策略定义为**条件逆向扩散过程**：从纯噪声 $a^K \sim \mathcal{N}(0, I)$ 出发，通过 $K$ 步迭代去噪，生成动作 $a^0 = a_t$：

$$a^K \to a^{K-1} \to \cdots \to a^1 \to a^0$$

每步去噪是一个**条件高斯分布**：

$$\pi_\theta(a^{k-1} | a^k, s) = \mathcal{N}\!\left(a^{k-1};\; \mu_\theta(a^k, s, k),\; \sigma_k^2 I\right) \tag{1}$$

其中均值由**ε-预测参数化**（见 [[wiki/concepts/DDPM]] §第三步）给出：

$$\mu_\theta(a^k, s, k) = \frac{1}{\sqrt{\alpha_k}}\!\left(a^k - \frac{\beta_k}{\sqrt{1-\bar{\alpha}_k}}\, \epsilon_\theta(a^k, s, k)\right) \tag{2}$$

采样时加入随机噪声：$a^{k-1} = \mu_\theta(a^k, s, k) + \sigma_k z$，$z \sim \mathcal{N}(0,I)$。

> **关键观察**：每步去噪的条件分布是**高斯分布**（式 1），这意味着对数似然 $\log \pi_\theta(a^{k-1}|a^k, s)$ 有**解析闭式表达**——这正是 DPPO 能够应用 PPO 的数学基础。

---

### 2.3 扩散策略的边缘分布问题

最终动作 $a^0$ 的边缘分布为：

$$\pi_\theta(a^0 | s) = \int p(a^K)\prod_{k=1}^{K} \pi_\theta(a^{k-1} | a^k, s)\; da^{1:K}$$

这个多重积分**没有解析解**！（类比变分自编码器中对隐变量的积分同样没有闭式。）

这正是为什么不能直接对 $\log \pi_\theta(a^0|s)$ 做策略梯度——这个量无法计算。

**DPPO 的核心创新**：用不同的方式重新定义"动作"，**绕开这个积分**。

---

## 3. DPPO 核心：双层 MDP 框架

### 3.1 直觉：把去噪过程变成一个 MDP

DPPO 的观点是：**不要把去噪过程看成"黑盒"，而是把它展开为一个独立的 MDP**。

```
┌─────────────────────────────────────────────────────────┐
│  外层 MDP（环境层）                                       │
│                                                         │
│  t=0: s_0  ──[经过内层去噪]──→  s_1, r_0                  │
│  t=1: s_1  ──[经过内层去噪]──→  s_2, r_1                  │
│  ...                                                    │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  内层 MDP（扩散层，在每个环境时刻 t 发生一次）          │  │
│  │                                                   │  │
│  │  k=K:  (s_t, a^K)    ──去噪──→  a^{K-1}           │  │
│  │  k=K-1:(s_t, a^{K-1})──去噪──→  a^{K-2}           │  │
│  │  ...                                              │  │
│  │  k=1:  (s_t, a^1)   ──去噪──→  a^0 = a_t          │  │
│  │                ↓                                  │  │
│  │         a_t 传入外层，获得 r_t                     │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**内层 MDP 的定义**：
- **状态**：$(s_t, a^k)$——观测 + 当前噪声动作
- **动作**：输出 $a^{k-1}$（即"决定如何去噪"）
- **奖励**：所有中间步均为 0，仅在 $k=0$ 时传递环境奖励 $r_t$

---

### 3.2 关键突破：用扩展轨迹定义策略

DPPO **不对隐变量 $a^{1:K-1}$ 积分**，而是将完整去噪轨迹 $(a^K, a^{K-1}, \ldots, a^0)$ 视为"扩展动作"。

**完整去噪轨迹的联合分布**（固定初始噪声 $a^K$ 和观测 $s$）：

$$\pi_\theta(a^{0:K-1} | a^K, s) = \prod_{k=1}^{K} \pi_\theta(a^{k-1} | a^k, s) \tag{3}$$

这个分解成立是因为：扩散过程是**马尔可夫链**——给定 $a^k$，$a^{k-1}$ 的分布只依赖 $a^k$ 和 $s$，与更早的状态无关。

取对数，乘积变求和：

$$\log \pi_\theta(a^{0:K-1} | a^K, s) = \sum_{k=1}^{K} \log \pi_\theta(a^{k-1} | a^k, s) \tag{4}$$

**这正是 DPPO 能够计算对数似然的关键**：虽然边缘分布 $\pi_\theta(a^0|s)$ 没有解析式，但**以完整轨迹为单位**的联合对数似然（式 4）是各步对数似然的简单求和！

---

## 4. 严谨数学推导

### 4.1 单步去噪的对数似然（Step-wise Log-likelihood）

**已知**：每步去噪分布是 $d$ 维高斯（式 1）：$\pi_\theta(a^{k-1}|a^k,s) = \mathcal{N}(a^{k-1};\, \mu_\theta(a^k,s,k),\, \sigma_k^2 I)$

**高斯分布的对数密度公式**（$d$ 维版本）：

$$\log \mathcal{N}(x;\, \mu,\, \sigma^2 I) = -\frac{d}{2}\log(2\pi) - d\log\sigma - \frac{\|x - \mu\|^2}{2\sigma^2}$$

**推导**：代入高斯密度函数 $p(x) = (2\pi\sigma^2)^{-d/2} \exp\!\left(-\frac{\|x-\mu\|^2}{2\sigma^2}\right)$，取对数即得。

因此，第 $k$ 步的对数似然为：

$$\log \pi_\theta(a^{k-1}|a^k,s) = \underbrace{-\frac{d}{2}\log(2\pi) - d\log\sigma_k}_{\text{与 }\theta\text{ 无关的常数}} - \underbrace{\frac{\|a^{k-1} - \mu_\theta(a^k,s,k)\|^2}{2\sigma_k^2}}_{\text{依赖 }\theta\text{ 的项}}$$

由于常数项对 $\theta$ 的梯度为 0，优化时可以丢弃。**有效对数似然**：

$$\log \pi_\theta(a^{k-1}|a^k,s) \propto -\frac{\|a^{k-1} - \mu_\theta(a^k, s, k)\|^2}{2\sigma_k^2} \tag{5}$$

**物理意义**：$a^{k-1}$ 距离预测均值 $\mu_\theta$ 越近，对数似然越高（即策略认为这条去噪路径越"合理"）。

---

### 4.2 重要性采样比的推导

**目标**：计算第 $k$ 步去噪的重要性采样比：

$$\rho_\theta^k = \frac{\pi_\theta(a^{k-1} | a^k, s)}{\pi_{\theta_{old}}(a^{k-1} | a^k, s)}$$

**Step 1**：取对数比：

$$\log \rho_\theta^k = \log \pi_\theta(a^{k-1}|a^k,s) - \log \pi_{\theta_{old}}(a^{k-1}|a^k,s)$$

**Step 2**：代入式 (5)（常数项相消）：

$$\log \rho_\theta^k = \left(-\frac{\|a^{k-1} - \mu_\theta(a^k,s,k)\|^2}{2\sigma_k^2}\right) - \left(-\frac{\|a^{k-1} - \mu_{\theta_{old}}(a^k,s,k)\|^2}{2\sigma_k^2}\right)$$

**Step 3**：合并：

$$\boxed{\log \rho_\theta^k = \frac{\|a^{k-1} - \mu_{\theta_{old}}(a^k,s,k)\|^2 - \|a^{k-1} - \mu_\theta(a^k,s,k)\|^2}{2\sigma_k^2}} \tag{6}$$

$$\rho_\theta^k = \exp\!\left(\log \rho_\theta^k\right) \tag{7}$$

**物理意义**：
- 若新均值 $\mu_\theta$ 比旧均值 $\mu_{\theta_{old}}$ **更接近**当前去噪样本 $a^{k-1}$，则分子 $> 0$，$\rho_\theta^k > 1$（新策略认为这个去噪步更可能）
- 若新均值**更远离** $a^{k-1}$，则 $\rho_\theta^k < 1$（新策略认为这个去噪步不太可能）

**重要优点**：$\rho_\theta^k$ 完全依赖均值的欧式距离，可以在推理时**解析计算**，无需任何数值积分！

---

### 4.3 完整 DPPO 目标函数

将 PPO-Clip（§1.4）应用到双层 MDP 的每个去噪步：

$$\boxed{L^{DPPO}(\theta) = \mathbb{E}_{t, k}\!\left[\min\!\left(\rho_\theta^k\, \hat{A}_t,\;\; \mathrm{clip}(\rho_\theta^k,\, 1{-}\epsilon_{clip},\, 1{+}\epsilon_{clip})\, \hat{A}_t\right)\right]} \tag{8}$$

**关键设计决策解析**：

**1. 为什么用单步比 $\rho_\theta^k$ 而非全局比 $\prod_k \rho_\theta^k$？**

若用完整轨迹的重要性采样比：
$$\rho_\theta^{full} = \prod_{k=1}^{K} \rho_\theta^k$$

当 $K = 100$ 时，即使每步只有微小偏差（如 $\rho_\theta^k = 1.01$），乘积会变成 $1.01^{100} \approx 2.7$——数值爆炸！反之，$0.99^{100} \approx 0.37$——数值消失。对每步独立 clip 可以避免这个问题。

**2. 为什么所有去噪步共用同一个优势 $\hat{A}_t$？**

奖励 $r_t$ 来自环境，在一次完整动作生成（全 $K$ 步去噪）完成后才获得。所有去噪步的"贡献"是联合的：每步是否"去噪得好"共同决定最终动作 $a^0$ 的质量。因此用统一的环境优势 $\hat{A}_t$ 来评价整个去噪过程是合理的。

**3. 为什么是对 $t$ 和 $k$ 同时取期望？**

目标函数等价于将所有 $(t, k)$ 组合的 clip 损失求平均，每个 $(t, k)$ 对应一个独立的"PPO 子问题"，通过梯度累积合并更新。

---

### 4.4 优势函数估计（GAE）

**目标**：在不知道真实 $Q^{\pi}(s_t, a_t)$ 的情况下，估计 $\hat{A}_t$。

**训练一个价值网络** $V_\phi(s_t)$，预测从 $s_t$ 出发的期望回报。

**TD 误差（Temporal Difference Error）**：

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \tag{9}$$

- $r_t + \gamma V_\phi(s_{t+1})$：Bellman 目标（用一步实际奖励 + 下一状态估值）
- $V_\phi(s_t)$：当前状态估值
- $\delta_t > 0$ 说明"实际回报比预期好"（好动作），$\delta_t < 0$ 说明"比预期差"（差动作）

**GAE（Generalized Advantage Estimation，Schulman 等 2016）**：

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l} = \delta_t + \gamma\lambda\, \delta_{t+1} + (\gamma\lambda)^2\, \delta_{t+2} + \cdots \tag{10}$$

**参数 $\lambda \in [0,1]$** 控制偏差-方差权衡：

| $\lambda$              | 等价于                                                             | 性质                       |
| ---------------------- | --------------------------------------------------------------- | ------------------------ |
| $\lambda = 0$          | 纯 TD：$\hat{A}_t = \delta_t$                                     | 低方差，高偏差（依赖 $V_\phi$ 的精度） |
| $\lambda = 1$          | 蒙特卡洛：$\hat{A}_t = \sum_{t'} \gamma^{t'-t} r_{t'} - V_\phi(s_t)$ | 低偏差，高方差（实际轨迹噪声大）         |
| $\lambda \approx 0.95$ | 两者折中                                                            | 实践中常用值                   |

**价值网络 $V_\phi$ 的训练目标**（独立于 $\pi_\theta$，最小化 MSE）：
$$L^{value}(\phi) = \mathbb{E}_t\!\left[\bigl(V_\phi(s_t) - \hat{R}_t\bigr)^2\right], \quad \hat{R}_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} \tag{11}$$

---

## 5. 关键工程技巧

### 5.1 截断去噪（Truncated Denoising）

**问题**：对全部 $K$ 步做 PPO，有效时域（inner MDP 步数）过长，梯度信号衰减严重。

**解决**：只对**最后 $K'$ 步**（$K' \ll K$，如 $K' = 10$）做策略优化，前 $K - K'$ 步保持旧策略不变。

**截断后的重要性采样比**（只对最后 $K'$ 步连乘）：

$$\rho_\theta^{trunc} = \prod_{k=1}^{K'} \rho_\theta^k \tag{12}$$

**截断后的目标函数**：

$$L^{DPPO-T}(\theta) = \mathbb{E}_{t}\!\left[\sum_{k=1}^{K'} \min\!\left(\rho_\theta^k\, \hat{A}_t,\; \mathrm{clip}(\rho_\theta^k, 1-\epsilon_{clip}, 1+\epsilon_{clip})\, \hat{A}_t\right)\right]$$

**直觉**：前 $K - K'$ 步相当于"固定的特征提取器"，最后 $K'$ 步相当于"可调节的策略头"。这与语言模型微调时只调最后几层类似。

---

### 5.2 DDIM 加速采样

**DDPM** 需要 $K = 100$ 步去噪；**DDIM**（Song 等 2020）可将步数减至 $K = 10 \sim 20$：

$$a^{k-1} = \sqrt{\bar{\alpha}_{k-1}}\underbrace{\!\left(\frac{a^k - \sqrt{1-\bar{\alpha}_k}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_k}}\right)}_{\hat{a}^0\text{（预测的干净动作）}} + \underbrace{\sqrt{1-\bar{\alpha}_{k-1} - \sigma_k^2}\,\epsilon_\theta}_{\text{方向分量}} + \underbrace{\sigma_k z}_{\text{随机项}} \tag{13}$$

只要 $\sigma_k > 0$，每步仍是**高斯转移**，DPPO 的所有推导完全适用。

采用 DDIM 的效果：
- 去噪步数 $K$：$100 \to 10$，每次采集数据时间减少 10 倍
- 截断技巧与 DDIM 正交，可叠加（如 DDIM $K=10$，截断 $K'=5$）

---

### 5.3 修改噪声调度（Modified Noise Schedule）

**问题**：在 RL 微调时，若 $\sigma_k$ 过小（策略近乎确定性），**缺乏探索**；若过大，动作质量差。

**DPPO 做法**：微调时使用比预训练更大的 $\sigma_k$，主动维持适度随机性。

**数学依据**：高斯策略 $\pi_\theta(a^{k-1}|a^k,s) = \mathcal{N}(\mu_\theta, \sigma_k^2 I)$ 的（差分）熵：

$$H[\pi_\theta(\cdot|a^k,s)] = \frac{d}{2}\ln(2\pi e\, \sigma_k^2) \tag{14}$$

$\sigma_k$ 越大，熵越高，探索范围越广。适当增大 $\sigma_k$ 防止策略过早坍缩到局部最优。

实践中，完整的 DPPO 训练损失还加入**熵正则项**：

$$L^{total}(\theta) = -L^{DPPO}(\theta) + c_1\, L^{value}(\phi) - c_2\, H[\pi_\theta] \tag{15}$$

其中 $c_1, c_2$ 是系数，$c_2 H[\pi_\theta]$ 项显式鼓励探索。

---

## 6. 结构化探索的数学直觉

### 6.1 高斯策略的探索方式

标准高斯策略：

$$\pi_{Gauss}(a|s) = \mathcal{N}(a;\, \mu_\theta(s),\, \sigma^2 I)$$

在动作空间 $\mathbb{R}^d$ 中，以 $\mu_\theta(s)$ 为中心**各向同性（isotropic）**地探索。

**问题**：机器人动作空间中，绝大多数方向物理上不可行（例如，机械臂的合法轨迹集合只占整个高维空间的极小子集）。高斯策略在 RL 微调时会频繁采样到物理上无意义的动作，导致大量"无效探索"。

---

### 6.2 扩散策略的结构化探索

设训练演示数据的分布支撑于低维流形 $\mathcal{M} \subset \mathbb{R}^d$（例如：所有合法机器人动作轨迹的集合）。

**DDPM 训练目标**（回忆 $L_{simple}$）：

$$L_{simple}(\theta) = \mathbb{E}_{a^0, k, \epsilon}\!\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_k}\, a^0 + \sqrt{1-\bar{\alpha}_k}\, \epsilon,\, s,\, k)\|^2\right]$$

通过最小化此目标，网络 $\epsilon_\theta$ 学会了近似**分数函数（Score Function）**：

$$\epsilon_\theta(a^k, s, k) \approx -\sqrt{1-\bar{\alpha}_k}\, \nabla_{a^k} \log q_k(a^k) \tag{16}$$

其中 $q_k$ 是 $k$ 步加噪后的边缘分布（由训练数据决定）。

**去噪更新**（代入式 16）：

$$a^{k-1} \approx a^k + \frac{\beta_k}{2}\, \nabla_{a^k} \log q_k(a^k) + \sigma_k z \tag{17}$$

这正是**退火朗之万动力学（Annealed Langevin Dynamics）**——每步都沿着数据分布的**对数梯度方向**移动，即朝向流形 $\mathcal{M}$ 方向移动。

**结论**：扩散策略的去噪过程在探索时，天然沿数据流形 $\mathcal{M}$ 移动，不会产生物理上不可能的样本。这就是"**结构化探索（Structured Exploration）**"。

---

### 6.3 量化对比

设流形 $\mathcal{M}$ 的本征维度（intrinsic dimension）为 $d_m \ll d$（例如 $d_m = 10$，$d = 100$）。

| 策略类型 | 探索体积 | 落在 $\mathcal{M}$ 附近的比例 |
|---------|---------|--------------------------|
| 高斯策略 $\mathcal{N}(\mu, \sigma^2 I)$ | $\propto \sigma^d$（$d$ 维球） | $\approx 0$（指数小）|
| 扩散策略 | $\propto \sigma_k^{d_m}$（沿流形的邻域）| $\approx 1$（结构化）|

扩散策略在流形附近的探索效率远高于高斯策略。这在实验中体现为：
- 微调时**策略对观测扰动更鲁棒**（因为探索集中在有意义的区域）
- **sim-to-real 迁移能力更强**（训练数据流形与真实数据流形相似）

---

## 7. 完整算法总结

```
算法：DPPO（含截断去噪 + DDIM）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入：预训练扩散策略参数 θ₀，价值网络参数 φ₀，
      DDIM 步数 K，截断步数 K'，裁剪范围 ε_clip，
      更新轮数 M（每批数据的 PPO 迭代次数）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
for 每次外循环迭代 i = 1, 2, ... do
  ─── 阶段 1：数据采集 ────────────────────────────
  令 θ_old = θ（保存当前策略作为旧策略）
  用 π_{θ_old} 与环境交互，收集 N 条轨迹：
    ● 对每个时刻 t：
        (1) 采样 a^K ~ N(0, I)（初始噪声）
        (2) DDIM 去噪 K 步，保存中间状态 a^{K-1}, ..., a^0
        (3) 执行动作 a^0，获得奖励 r_t，转移到 s_{t+1}

  ─── 阶段 2：优势估计 ───────────────────────────
  对每条轨迹计算 GAE 优势 Â_t（式 10）
  和折扣回报 R̂_t（用于训练价值网络）

  ─── 阶段 3：PPO 更新（迭代 M 轮）───────────────
  for epoch = 1, ..., M do
    for 每个 mini-batch (s_t, a^{0:K}_t, r_t, Â_t) do

      ── 策略损失（式 8）──
      for k = 1, ..., K' do            ← 只更新最后 K' 步
        计算 μ_θ(a^k, s, k)（用当前网络）
        计算 μ_{θ_old}(a^k, s, k)（用旧网络）
        计算 log ρ_θ^k = (‖a^{k-1}-μ_{θ_old}‖² - ‖a^{k-1}-μ_θ‖²) / (2σ_k²)   [式 6]
        ρ_θ^k = exp(log ρ_θ^k)         [式 7]
        L_k = min(ρ_θ^k · Â_t, clip(ρ_θ^k, 1-ε_clip, 1+ε_clip) · Â_t)   [式 8]
      end for
      L_policy = -mean(L_k)             ← 取负号因为是最大化

      ── 价值损失（式 11）──
      L_value = mean((V_φ(s_t) - R̂_t)²)

      ── 总损失（式 15）──
      L_total = L_policy + c₁ · L_value - c₂ · H[π_θ]

      ── 梯度更新 ──
      (θ, φ) ← (θ, φ) - η · ∇_{θ,φ} L_total
    end for
  end for

end for
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出：微调后的扩散策略 π_θ
```

---

## 8. 关键公式速查

| 编号   | 公式                                                                                                                                 | 含义            |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| (1)  | $\pi_\theta(a^{k-1}\|a^k,s) = \mathcal{N}(a^{k-1};\,\mu_\theta(a^k,s,k),\,\sigma_k^2 I)$                                           | 单步去噪的条件高斯分布   |
| (2)  | $\mu_\theta(a^k,s,k) = \frac{1}{\sqrt{\alpha_k}}\!\left(a^k - \frac{\beta_k}{\sqrt{1-\bar{\alpha}_k}}\epsilon_\theta\right)$       | ε-预测参数化（DDPM） |
| (4)  | $\log\pi_\theta(a^{0:K-1}\|a^K,s) = \sum_{k=1}^K \log\pi_\theta(a^{k-1}\|a^k,s)$                                                   | 马尔可夫分解：乘积→求和  |
| (5)  | $\log\pi_\theta(a^{k-1}\|a^k,s) \propto -\frac{\|a^{k-1}-\mu_\theta\|^2}{2\sigma_k^2}$                                             | 对数似然的有效项      |
| (6)  | $\log\rho_\theta^k = \frac{\|a^{k-1}-\mu_{\theta_{old}}\|^2 - \|a^{k-1}-\mu_\theta\|^2}{2\sigma_k^2}$                              | 单步 IS 比（解析可算） |
| (8)  | $L^{DPPO} = \mathbb{E}_{t,k}\!\left[\min(\rho_\theta^k\hat{A}_t,\,\text{clip}(\rho_\theta^k,1\pm\epsilon_{clip})\hat{A}_t)\right]$ | DPPO 核心目标函数   |
| (10) | $\hat{A}_t^{GAE} = \sum_{l\geq 0}(\gamma\lambda)^l\delta_{t+l}$                                                                    | GAE 优势估计      |

---

## 9. 为什么 DPPO 有效：三个关键问题

**Q1：之前的工作说"策略梯度对扩散策略无效"，DPPO 如何反驳？**

先前 Psenka 等推测：去噪步数 $K$ 放大了有效时域，导致梯度消失，PG 无效。DPPO 通过两点解决：
1. **截断去噪**（§5.1）：将有效时域从 $K$ 缩短到 $K'$
2. **单步 clip**（§4.3）：每步独立 clip，避免连乘导致的爆炸/消失

实验上 DPPO 在所有基线中综合表现最好，直接否定了先前推测。

**Q2：为什么不用更简单的 REINFORCE？**

REINFORCE 是高方差的"蒙特卡洛"估计，在长时序稀疏奖励任务（如多阶段桌面装配）中方差极大，训练极不稳定。PPO 的 GAE + 价值网络显著降低了方差，使长任务上的训练可行。

**Q3：为什么扩散策略比高斯策略微调效果更好？**

结构化探索（§6）：扩散策略在数据流形附近探索，高斯策略在整个动作空间各向同性探索。前者探索更"有方向"，微调时能找到更好的动作，而不会陷入物理上无意义的区域。这在实验中体现为显著的 sim-to-real 迁移能力优势。
