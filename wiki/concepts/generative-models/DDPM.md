---
type: concept
tags: [扩散模型, 生成模型, NeurIPS, DDPM, 去噪分数匹配, Langevin动力学]
sources: [raw/sources/papers/Generative Model/Ho 等 - 2020 - Denoising Diffusion Probabilistic Models.pdf, raw/sources/papers/IL(Imitation Learning)/Chi 等 - 2024 - Diffusion Policy Visuomotor Policy Learning via Action Diffusion.pdf, raw/sources/papers/VLA+RL/Ren 等 - 2024 - Diffusion Policy Policy Optimization.pdf]
created: 2026-04-21
updated: 2026-04-21
---

# DDPM (Denoising Diffusion Probabilistic Models)

**DDPM**（Ho et al., NeurIPS 2020）是现代扩散生成模型的奠基性框架，通过参数化马尔可夫链的**逆向去噪过程**来生成数据。它揭示了扩散模型与去噪分数匹配、退火 Langevin 动力学之间的深刻等价关系，并确立了 **ε-预测参数化**和**简化训练目标 $L_{simple}$** 作为事实标准。

---

## 核心直觉

扩散模型的思路分两步：
1. **前向过程**：固定的马尔可夫链，逐步向数据加噪，直到变为纯高斯噪声。
2. **逆向过程**：学习一个神经网络，逐步从噪声中恢复数据——即"学会逆向时间轴"。

推理时，从 $x_T \sim \mathcal{N}(0, I)$ 出发，反复执行逆向步骤，最终得到 $x_0 \sim p_{data}$。

---

## 严谨数学推导

### 符号定义

| 符号 | 含义 |
|------|------|
| $x_0 \sim q(x_0)$ | 真实数据分布 |
| $x_1, \ldots, x_T$ | 隐变量（加噪后的中间状态），维度与 $x_0$ 相同 |
| $\beta_t \in (0,1)$ | 前向过程方差调度（variance schedule），固定超参数 |
| $\alpha_t = 1 - \beta_t$ | 每步信号保留比例 |
| $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ | 累积信号保留比例 |
| $T$ | 总扩散步数（DDPM 取 $T=1000$）|
| $\epsilon_\theta(x_t, t)$ | 神经网络（U-Net），预测噪声 $\epsilon$ |

---

### 第一步：前向过程（Forward Process）

定义固定的马尔可夫加噪链：

$$
q(x_{1:T} \mid x_0) := \prod_{t=1}^{T} q(x_t \mid x_{t-1}), \qquad
q(x_t \mid x_{t-1}) := \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t \mathbf{I}\right)
$$

**物理意义**：每步将 $x_{t-1}$ 缩放 $\sqrt{1-\beta_t}$（保留信号）再叠加标准差为 $\sqrt{\beta_t}$ 的高斯噪声。

**Closed-form 采样**（关键推导）：

利用高斯分布的再生性，可以跳过中间步骤，直接从 $x_0$ 采样任意时刻 $x_t$：

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

即：

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) \mathbf{I}\right) \tag{4}
$$

**推导过程**（归纳法）：

- $t=1$：$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1}\, \epsilon_1$，因此 $\bar{\alpha}_1 = \alpha_1$，成立。
- 若 $x_{t-1} \mid x_0 \sim \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} x_0,\, (1-\bar{\alpha}_{t-1})I)$，则：
  $$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t}\, \epsilon_t$$
  均值为 $\sqrt{\alpha_t} \cdot \sqrt{\bar{\alpha}_{t-1}} x_0 = \sqrt{\bar{\alpha}_t} x_0$，
  方差为 $\alpha_t (1-\bar{\alpha}_{t-1}) + (1-\alpha_t) = 1 - \bar{\alpha}_t$。命题成立。

---

### 第二步：变分下界（ELBO）

极大化对数似然 $\log p_\theta(x_0)$ 等价于最小化其 ELBO 的负值。利用重要性采样，可将变分下界分解为各时间步的 KL 散度之和：

$$
\mathbb{E}_q\!\left[-\log p_\theta(x_0)\right] \leq \mathbb{E}_q\!\left[\underbrace{D_{\mathrm{KL}}\!\left(q(x_T \mid x_0) \| p(x_T)\right)}_{L_T} + \sum_{t>1}\underbrace{D_{\mathrm{KL}}\!\left(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t)\right)}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 \mid x_1)}_{L_0}\right] \tag{5}
$$

**关键技巧**：前向过程的后验 $q(x_{t-1} \mid x_t, x_0)$ 在 $x_0$ 条件下是 tractable 高斯：

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\; \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t \mathbf{I}\right)
$$

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\, x_t, \qquad
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\, \beta_t
$$

因此每个 $L_{t-1}$ 项为两个高斯之间的 KL 散度，有 closed-form 解析表达式，无需高方差 Monte Carlo 估计。

---

### 第三步：ε-预测参数化

令 $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$，并将式 (4) 代入 $L_{t-1}$：

$$
L_{t-1} - C = \mathbb{E}_{x_0,\epsilon}\!\left[\frac{1}{2\sigma_t^2}\left\|\frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right) - \mu_\theta(x_t, t)\right\|^2\right]
$$

**洞察**：括号内恰好是 $\tilde{\mu}_t$ 的"噪声参数化"形式。定义网络预测噪声：

$$
\mu_\theta(x_t, t) := \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\right)
$$

则 $L_{t-1}$ 化简为：

$$
\mathbb{E}_{x_0,\epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}\, \left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\; t\right)\right\|^2\right] \tag{12}
$$

**等价关系**：式 (12) 正是对噪声级别 $t$ 的**去噪分数匹配（Denoising Score Matching）**目标——训练神经网络预测添加的噪声，等价于估计数据分布的对数梯度（score）。

---

### 第四步：简化目标 $L_{simple}$

丢弃式 (12) 中时间步依赖的加权系数，得到等权 MSE：

$$
L_{simple}(\theta) := \mathbb{E}_{t \sim \mathrm{Uniform}[1,T],\; x_0,\; \epsilon}\!\left[\left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\; t\right)\right\|^2\right]
$$

**为何更好**：原权重在小 $t$ 时极大（网络做几乎无噪声的去噪，容易但无益），$L_{simple}$ 下调了这些项的权重，迫使网络专注于困难的大噪声场景，采样质量更高。

---

### 第五步：采样（退火 Langevin 动力学）

从 $x_T \sim \mathcal{N}(0, I)$ 出发，反复执行：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})
$$

此更新规则在形式上等同于以 $\epsilon_\theta$ 为数据分布对数梯度近似的**退火 Langevin 动力学采样**。

---

## 完整数学推导

> 本概念页仅做概要介绍。如需从零推导所有公式（包括后验配方、ELBO 分解、ε-参数化、DDIM 加速采样等），请参见：
> **[[wiki/analyses/DDPM & DDIM 完整数学推导]]**

---

## 与相关方法的联系

| 方法 | 关系 |
|------|------|
| [[wiki/concepts/generative-models/Diffusion Policy]] | DDPM 应用于机器人动作生成（Chi et al., RSS 2023）；逆向过程预测动作序列而非图像 |
| [[wiki/concepts/generative-models/Flow Matching]] | 替代路线：用 ODE 流代替扩散马尔可夫链，轨迹更直，推理步数更少 |
| [[wiki/concepts/rl/DPPO]] | 用 PPO 对 DDPM-based Diffusion Policy 做在线 RL 微调 |
| NCSN（Song & Ermon, 2019）| 同期分数匹配工作；DDPM 通过变分推断直接训练采样器，更严谨 |

---

## 在知识库中的出现

| 来源 | 角色 |
|------|------|
| [[wiki/sources/generative/2026-04-21 DDPM (Ho 2020)]] | **原始论文**：方法推导、CIFAR10/LSUN 实验 |
| [[wiki/sources/imitation-learning/2026-04-19 Diffusion Policy (Chi 2024)]] | DDPM 作为机器人策略的生成骨干 |
| [[wiki/sources/rl-finetuning/2026-04-19 DPPO]] | DDPM-based 策略的 RL 微调对象 |
