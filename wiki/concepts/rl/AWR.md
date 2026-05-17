---
type: concept
tags: [Offline RL, Advantage Weighting, Imitation Learning, Policy Optimization, Behavior Cloning]
sources: [π₀.₆ 论文 (Intelligence 等 2025), Flow Q-Learning (Park 等 2025)]
created: 2026-05-17
updated: 2026-05-17
---

# AWR — Advantage-Weighted Regression

AWR（Advantage-Weighted Regression，优势加权回归）是 Peng et al. (2019) 提出的一种经典 **offline RL** 算法。其核心思想是：**不做策略梯度，而是用 advantage 对监督学习样本进行重新加权**，从而在行为克隆基础上隐式地实现策略改进。

**关联页面**：[[wiki/concepts/rl/Offline 强化学习]] | [[wiki/concepts/rl/RECAP]] | [[wiki/analyses/π₀.₆ 与 RECAP 训练原理全景解析]]

---

## 核心思想

AWR 的出发点是 KL 正则化 RL 的优化目标：

$$
\mathcal{J}(\pi, \pi_{\mathrm{ref}}) = \mathbb{E}_{\tau \sim \rho_\pi}\!\left[\sum_{t} \gamma^t r_t\right] - \beta\, \mathrm{KL}\!\left(\pi(\cdot \mid o) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid o)\right)
$$

对此目标用 Lagrangian 方法求解，得到**闭式最优策略**：

$$
\boxed{\hat\pi(a \mid o) \propto \pi_{\mathrm{ref}}(a \mid o) \cdot \exp\!\left(\frac{A^{\pi_{\mathrm{ref}}}(o, a)}{\beta}\right)}
$$

其中 $A^{\pi_{\mathrm{ref}}}(o,a) = Q(o,a) - V(o)$ 是优势函数，$\beta > 0$ 是 KL 正则化强度。

---

## 严谨数学推导

### 符号定义

| 符号                             | 含义                   |
| ------------------------------ | -------------------- |
| $\pi_{\mathrm{ref}}(a \mid o)$ | 参考策略（行为策略 / 上一轮策略）   |
| $\hat\pi(a \mid o)$            | 改进策略（待求）             |
| $A^{\pi_{\mathrm{ref}}}(o, a)$ | 优势函数：$Q(o,a) - V(o)$ |
| $\beta$                        | KL 散度正则化强度（温度参数）     |
| $\mathcal{D}$                  | 离线数据集                |

### 推导过程

**起点**：最大化 KL 正则化目标 $\mathcal{J}$。

**Step 1 — 用拉格朗日对偶将约束吸收进目标**

对每个状态 $o$，最优动作分布满足：

$$
\hat\pi(a \mid o) = \arg\max_\pi \mathbb{E}_{a \sim \pi}\left[Q(o,a)\right] - \beta\, \mathrm{KL}(\pi \| \pi_{\mathrm{ref}})
$$

*这是逐状态的凸优化，可对 $\pi(a|o)$ 求解析解。*

**Step 2 — 变分法求解**

令 $\mathcal{L} = \sum_a \pi(a)\, Q(a) - \beta \sum_a \pi(a)\ln\frac{\pi(a)}{\pi_{\mathrm{ref}}(a)} - \lambda(\sum_a \pi(a) - 1)$，对 $\pi(a)$ 求导并令其为零：

$$
Q(a) - \beta\!\left(\ln\frac{\pi(a)}{\pi_{\mathrm{ref}}(a)} + 1\right) - \lambda = 0
$$

整理得：

$$
\hat\pi(a \mid o) = \frac{1}{Z(o)}\, \pi_{\mathrm{ref}}(a \mid o) \cdot \exp\!\left(\frac{Q(o,a)}{\beta}\right)
$$

*$Z(o) = \int \pi_{\mathrm{ref}}(a'\mid o) \exp(Q(o,a')/\beta)\, da'$ 是归一化常数。*

**Step 3 — 用 Advantage 替换 Q（去掉归一化常数的依赖）**

由于 $\exp(Q/\beta) = \exp\!\left((A + V)/\beta\right) = \exp(V(o)/\beta) \cdot \exp(A/\beta)$，而 $\exp(V(o)/\beta)$ 对所有动作相同，可并入 $Z(o)$ 消去，得：

$$
\hat\pi(a \mid o) \propto \pi_{\mathrm{ref}}(a \mid o) \cdot \exp\!\left(\frac{A^{\pi_{\mathrm{ref}}}(o, a)}{\beta}\right)
$$

*物理意义：高 advantage 的动作获得更高权重，低 advantage 的动作权重趋近于 0。*

**Step 4 — 实践：把闭式解变成有监督学习**

无法直接从 $\hat\pi$ 采样，但可以对现有数据集 $\mathcal{D}$ 做**加权行为克隆**：

$$
\mathcal{L}_{\mathrm{AWR}}(\theta) = -\mathbb{E}_{(o,a) \sim \mathcal{D}}\!\left[\underbrace{\exp\!\left(\frac{A(o,a)}{\beta}\right)}_{w}\cdot \log \pi_\theta(a \mid o)\right]
$$

*$A(o,a) > 0$：权重 $w > 1$，多学；$A(o,a) < 0$：权重 $w < 1$，少学。*

---

## 本质：过滤式模仿学习

AWR 的本质是**过滤式模仿学习（Filtered Imitation Learning）**：

```
普通行为克隆：  所有样本等权重 BC
AWR：          advantage > 0 的样本多学，advantage < 0 的样本少学（但不彻底丢弃）
```

与其他 offline RL 方法对比：

| 方法                             | 核心机制                       | 负样本处理                      |
| ------------------------------ | -------------------------- | -------------------------- |
| **AWR**                        | 加权 BC，权重 $= \exp(A/\beta)$ | 权重趋近 0，几乎忽略                |
| CQL / IQL                      | 惩罚 OOD 动作的 Q 值             | 显式惩罚                       |
| **[[wiki/concepts/rl/RECAP]]** | 二值化条件 $I$，条件 BC            | 作为 $I=\text{neg}$ **主动学习** |
| Decision Transformer           | Return-to-go 作为条件          | 无显式惩罚                      |

---

## 局限性

AWR 在大型模型（VLA / 扩散策略）上的核心问题：

| 问题 | 原因 |
|---|---|
| **数值不稳定** | $A$ 较大时 $\exp(A/\beta)$ 数值爆炸 |
| **梯度方差高** | 极端权重主导梯度，等价于高方差估计 |
| **Flow Matching 不兼容** | $\exp(A/\beta)$ 无法直接乘到 CFM loss 前，会破坏 $\eta$-期望结构 |
| **差样本浪费** | 低 advantage 样本权重 $\to 0$，学习信号消失 |
| **速度下降** | 仅对高 advantage 样本学习，忽略"快且成功"的次优动作，策略变慢 |

这些局限性是 [[wiki/concepts/rl/RECAP]] 提出 advantage conditioning（以 $I$ 替代 $\exp(A/\beta)$）的核心动机。

---

## 实验表现（π₀.₆ T-shirt 折叠任务）

来自 [[wiki/analyses/π₀.₆ 与 RECAP 训练原理全景解析]]（Figure 11）：

- **成功率**：AWR ≈ offline-RL+SFT，显著优于 PPO
- **吞吐量**：AWR 策略**明显变慢**——因为 AWR 只学快且成功的动作，牺牲了速度
- RECAP 同时在成功率和速度上超过 AWR

---

## 出现来源

- [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] — RECAP vs AWR 的工程比较（Figure 11，T-shirt 任务）
- [[wiki/concepts/rl/Offline 强化学习]] — AWR 被列为"加权回归"一类的代表方法
- 原始论文：Peng et al., "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning", arXiv 2019
