---
type: analysis
tags: [ReinFlow, Flow Matching, 在线强化学习, Policy Gradient, 噪声注入, Markov Process, PPO, 机器人控制, NeurIPS 2025]
sources: [raw/sources/papers/VLA+RL/Zhang 等 - 2026 - ReinFlow Fine-tuning Flow Matching Policy with Online Reinforcement Learning_1.md]
created: 2026-04-24
updated: 2026-04-24
---

# ReinFlow $\pi_{rl}$ 完整数学推导

> **来源论文**：[[wiki/sources/2026-04-18 ReinFlow|ReinFlow: Fine-tuning Flow Matching Policy with Online RL]]（Zhang 等，NeurIPS 2025）
> **分析目标**：从头推导 ReinFlow 如何将 Flow Matching 策略转化为可被在线 RL 优化的策略 $\pi_{rl}$，并给出 Policy Gradient Theorem 4.1 的完整证明。

---

## 0. 符号定义

| 符号                                      | 含义                                                          |
| --------------------------------------- | ----------------------------------------------------------- |
| $\mathcal{S}, \mathcal{A}, \mathcal{O}$ | 状态空间、动作空间、观测空间                                              |
| $s_h, o_h, a_h$                         | 第 $h$ 步的状态、观测、动作                                            |
| $\gamma \in (0,1)$                      | 折扣因子                                                        |
| $r_h(o_h, a_h)$                         | 即时奖励                                                        |
| $J(\pi)$                                | 期望折扣累积奖励（目标函数）                                              |
| $Q_h^\pi, V_h^\pi, A_h^\pi$             | Q 函数、价值函数、优势函数                                              |
| $v_\theta(t, a, o)$                     | 预训练 Flow 策略的速度网络（参数 $\theta$，**训练后保留**）                     |
| $\sigma_{\theta'}(t, a, o)$             | 噪声注入网络（参数 $\theta'$，**训练后丢弃**）                              |
| $\bar\theta = [\theta, \theta']$        | 联合参数                                                        |
| $a^k$                                   | 去噪链第 $k$ 步的中间动作（$a^0 \sim \mathcal{N}(0,I)$，$a^K = a$ 执行动作） |
| $K$                                     | 总去噪步数                                                       |
| $t_k = k/K,\; \Delta t_k = 1/K$         | 均匀时间离散化                                                     |
| $\pi_{bc}$                              | 行为克隆（预训练）流策略                                                |
| $\pi_{rl}$                              | RL 微调后的策略（边缘分布）                                             |
| $d_\rho^\pi$                            | 折扣观测访问频率                                                    |

---

## 1. 问题背景：为什么 Flow 策略难以做在线 RL？

### 1.1 POMDP 框架

将机器人控制建模为**无限视界 POMDP**，目标是最大化：

$$
J(\pi) = \mathbb{E}^\pi\!\left[\sum_{h=0}^{+\infty} \gamma^h r_h(o_h, a_h)\right]
$$

Q 函数、价值函数与优势函数定义为：

$$
Q_h^\pi(o_h, a_h) := \mathbb{E}^\pi\!\left[\sum_{\tau=h}^{+\infty}\gamma^{\tau-h}r_\tau \;\middle|\; o_h, a_h\right], \quad
V_h^\pi(o_h) := \mathbb{E}_{a_h}\!\left[Q_h^\pi(o_h,a_h)\right], \quad
A_h^\pi := Q_h^\pi - V_h^\pi
$$

### 1.2 Flow Matching 策略的基本形式

**Flow Matching**（[[wiki/concepts/Flow Matching|概念页]]）学习一个速度场 $v_\theta$，使 ODE

$$
\frac{d}{dt}\psi_t(a^0) = v_\theta\!\left(t,\, \psi_t(a^0),\, o\right), \qquad a^0 \sim \mathcal{N}(0, I_{d_A})
$$

将高斯噪声映射到动作分布。其中 **Rectified Flow** 采用线性插值路径 $a_t = (1-t)a^0 + t\,a^1$，训练目标为：

$$
\hat\theta = \arg\min_\theta\; \mathbb{E}_{a^0 \sim \mathcal{N}(0,I),\; a^1 \sim p_1,\; t \sim \mathrm{Unif}[0,1]} \left[\|a^1 - a^0 - v_\theta(t, a_t, o)\|_2^2\right] \tag{Eq.2}
$$

**Shortcut Models** 在此基础上，强制两步生成的速度与精细步一致，进一步提升少步推理质量。

### 1.3 两大技术障碍

在线 RL 依赖**策略梯度** $\nabla_\theta \ln\pi(a|o)$，而 Flow 策略有两个致命缺陷：

| 问题             | 根因                               | 后果                            |
| -------------- | -------------------------------- | ----------------------------- |
| **Log 概率不可解析** | ODE 路径确定，转移概率为 Dirac $\delta$ 函数 | 无法直接计算 $\ln\pi_{rl}(a\mid o)$ |
| **缺乏探索机制**     | 确定性 ODE 无随机性                     | 无法在稀疏奖励任务中有效探索                |

连续时间精确解 $\ln p_1(\psi_1(x)) = \ln p_0(x) - \int_0^1 \nabla\cdot v(t,\psi_t(x))\,dt$ 需要蒙特卡洛估计散度，步数少时离散误差大，且计算昂贵。

---

## 2. $\pi_{rl}$ 的构造：噪声注入将 ODE 转化为 Markov 过程

### 2.1 噪声注入机制（核心贡献）

ReinFlow 在每个去噪步注入**可学习的高斯噪声**：

$$
\boxed{
a^0 \sim \mathcal{N}(0, I_{d_A}), \qquad
a^{k+1} \sim \mathcal{N}\!\left(\,a^k + v_\theta(t_k, a^k, o)\,\Delta t_k,\;\; \sigma_{\theta'}^2(t_k, a^k, o)\,\right)
} \tag{Eq.6}
$$

**物理意义**：
- 均值 $a^k + v_\theta \Delta t_k$ 是原始 ODE 的 Euler 步（保留预训练知识）
- $\sigma_{\theta'}(t_k, a^k, o)$ 是以 $(t_k, a^k, o)$ 为条件的**可学习标准差**，条件化保持 Markov 性质

这一设计将原本确定性的 ODE 路径变为**离散时间马尔可夫过程（Discrete-time Markov Process）**，其联合分布为：

$$
\pi^{\bar\theta}(a^0, a^1, \ldots, a^K \mid o) = \mathcal{N}(a^0;\, 0, I) \cdot \prod_{k=0}^{K-1} \mathcal{N}\!\left(a^{k+1};\; a^k + v_\theta\Delta t_k,\; \sigma_{\theta'}^2\right)
$$

**$\pi_{rl}$ 的定义**：RL 优化的目标策略是去噪链的**边缘动作分布**：

$$
\pi_{rl}(a \mid o) = \pi^{\bar\theta}(a^K \mid o) = \int_{\mathcal{A}^K} \pi^{\bar\theta}(a^0, \ldots, a^{K-1}, a^K \mid o)\; da^0 \cdots da^{K-1}
$$

### 2.2 封闭形式 Log 概率（Eq. 7）

由去噪链的马尔可夫因子分解，联合 log 概率为：

$$
\boxed{
\ln\pi\!\left(a^0,\ldots,a^K \mid o;\, \theta,\theta'\right) = \underbrace{\ln\mathcal{N}(a^0;\,0,I)}_{\text{初始噪声}} + \sum_{k=0}^{K-1} \underbrace{\ln\mathcal{N}\!\left(a^{k+1} \;\middle|\; a^k + v_\theta(t_k,a^k,o)\Delta t_k,\; \sigma_{\theta'}^2(t_k,a^k,o)\right)}_{\text{第 }k\text{ 步的高斯转移概率}}
} \tag{Eq.7}
$$

其中每一项高斯对数概率展开为：

$$
\ln\mathcal{N}(a^{k+1} \mid \mu_k, \sigma_k^2) = -\frac{d_A}{2}\ln(2\pi) - d_A\ln\sigma_k - \frac{\|a^{k+1} - \mu_k\|_2^2}{2\sigma_k^2}
$$

**关键优势**：
- **精确，无近似**：对任意步数 $K$（包括 $K=1$）均无离散化误差
- **无蒙特卡洛**：不需要 Hutchinson 迹估计
- **可直接反传**：对 $\theta, \theta'$ 均可微

---

## 3. 严谨数学推导：Markov Process Policy Gradient Theorem

> **定理 4.1**（Markov Process PG 定理）：对于由离散时间 Markov 过程参数化的响应式策略 $\pi^\theta$，策略梯度为：
>
> $$\nabla_\theta J(\pi^\theta) = \mathbb{E}^{\pi^\theta}\!\left[\sum_{h=0}^{+\infty}\gamma^h A_h^{\pi^\theta}(o_h, a_h)\,\nabla_\theta \ln\pi^\theta(a_h^0, a_h^1, \ldots, a_h^K \mid o_h)\right] \tag{Eq.8}$$
>
> 当 POMDP 和策略均平稳时：
>
> $$\nabla_\theta J(\pi^\theta) = \frac{1}{1-\gamma}\mathbb{E}_{o \sim d_\rho^{\pi^\theta}}\mathbb{E}_{a^0,\ldots,a^K \sim \pi^\theta(\cdot|o)}\!\left[A^{\pi^\theta}(o,a)\;\nabla_\theta\sum_{k=0}^{K-1}\ln\pi^\theta(a^{k+1}\mid a^k, o)\right] \tag{Eq.9}$$

### 3.1 第一阶段：标准策略梯度（对边缘动作策略）

**目标**：证明 $\nabla_\theta J(\pi^\theta) = \mathbb{E}^{\pi^\theta}\left[\sum_{\tau=0}^{+\infty}\gamma^\tau A_\tau^{\pi^\theta}(o_\tau,a_\tau)\,\nabla_\theta\ln\pi_\theta(a_\tau \mid o_\tau)\right]$

**证明**（POMDP 上的策略梯度定理）：

$$
\nabla_\theta J(\pi^\theta) = \sum_{h=0}^{+\infty}\gamma^h \int_{\mathcal{O}\times\mathcal{A}^h} r_h(o_h,a_h) \cdot \nabla_\theta p(o_{1:h},a_{1:h} \mid \pi^\theta)
$$

由**对数导数技巧** $\nabla_\theta f = f \cdot \nabla_\theta \ln f$：
$$
\nabla_\theta J(\pi^\theta) = \sum_{h=0}^{+\infty}\gamma^h \mathbb{E}^{\pi^\theta}\!\left[r_h(o_h,a_h)\cdot\nabla_\theta\ln p(o_{1:h},a_{1:h}\mid\pi^\theta)\right]
$$

由 POMDP 的马尔可夫性（转移核和观测核不依赖 $\theta$）：

$$
\nabla_\theta\ln p(o_{1:t},a_{1:t}\mid\pi^\theta) = \sum_{\tau=1}^{t}\nabla_\theta\ln\pi_\theta(a_\tau \mid o_\tau) \tag{Eq.15}
$$

*推导*：轨迹概率分解为 $p = \rho(s_1)\prod_\tau \mathbb{O}(o_\tau|s_\tau)\pi_\theta(a_\tau|o_\tau)\mathbb{T}(s_{\tau+1}|s_\tau,a_\tau)$，取对数后只有 $\pi_\theta$ 项含 $\theta$。

代入后**交换求和顺序**（$h \ge \tau$ → $\tau \le h$）：

$$
= \mathbb{E}^{\pi^\theta}\!\left[\sum_{\tau=0}^{+\infty}\gamma^\tau\nabla_\theta\ln\pi_\theta(a_\tau|o_\tau)\sum_{h=\tau}^{+\infty}\gamma^{h-\tau}r_h(o_h,a_h)\right]
= \mathbb{E}^{\pi^\theta}\!\left[\sum_{\tau=0}^{+\infty}\gamma^\tau Q_\tau^{\pi^\theta}(o_\tau,a_\tau)\,\nabla_\theta\ln\pi_\theta(a_\tau|o_\tau)\right]
$$

最后减去零项（基线 $V$，对 $a_\tau$ 积分后梯度为 0）：

$$
\nabla_\theta J(\pi^\theta) = \mathbb{E}^{\pi^\theta}\!\left[\sum_{\tau=0}^{+\infty}\gamma^\tau A_\tau^{\pi^\theta}(o_\tau,a_\tau)\,\nabla_\theta\ln\pi_\theta(a_\tau\mid o_\tau)\right] \tag{Eq.14}
$$

### 3.2 第二阶段：扩展至 Markov Process 参数化策略

当 $a_\tau = a_\tau^K$ 由 Markov 链 $a_\tau^0 \to a_\tau^1 \to \cdots \to a_\tau^K$ 生成时，边缘概率为：

$$
\pi_\theta(a_h \mid o_h) = \int_{\mathcal{A}^K} \pi_\theta(a_h^0 \mid o_h)\cdot\prod_{k=0}^{K-1}\pi_\theta(a_h^{k+1}\mid a_h^k, o_h)\;da_h^0\cdots da_h^{K-1} \tag{Eq.16}
$$

将 Eq. (14) 中的 $\nabla_\theta\ln\pi_\theta(a_\tau|o_\tau)$ 展开：

$$
\nabla_\theta J = \sum_{\tau=0}^{+\infty}\gamma^\tau\int_\mathcal{A} da_\tau^K\; A_\tau(o_\tau,a_\tau)\cdot\nabla_\theta\int_{\mathcal{A}^K} da_\tau^0\cdots da_\tau^{K-1}\;\pi_\theta(a_\tau^0,\ldots,a_\tau^K \mid o_\tau)
$$

再次应用对数导数技巧和联合分布的因子分解：

$$
\nabla_\theta\ln\pi_\theta(a^0,\ldots,a^K \mid o) = \nabla_\theta\left[\ln\pi_\theta(a^0\mid o) + \sum_{k=0}^{K-1}\ln\pi_\theta(a^{k+1}\mid a^k, o)\right]
$$

注意 $\ln\pi_\theta(a^0\mid o) = \ln\mathcal{N}(a^0;0,I)$ 与 $\theta$ 无关（初始分布固定），因此：

$$
\nabla_\theta\ln\pi_\theta(a^0,\ldots,a^K\mid o) = \nabla_\theta\sum_{k=0}^{K-1}\ln\pi_\theta(a^{k+1}\mid a^k, o)
$$

整理得（非平稳形式，Eq. 8 / Eq. 17）：

$$
\nabla_\theta J(\pi^\theta) = \mathbb{E}^{\pi^\theta}\!\left[\sum_{\tau=0}^{+\infty}\gamma^\tau A_\tau^{\pi^\theta}(o_\tau,a_\tau)\,\nabla_\theta\sum_{k=0}^{K-1}\ln\pi_\theta(a_\tau^{k+1}\mid a_\tau^k, o_\tau)\right]
$$

### 3.3 第三阶段：平稳 POMDP 的重要性引理（Eq.19）

**引理**：对任意函数 $f:\mathcal{O}\times\mathcal{A}\to\mathbb{R}$，响应式平稳策略 $\pi$，有：

$$
\mathbb{E}^\pi\!\left[\sum_{h=0}^{+\infty}\gamma^h f(o_h,a_h)\right] = \frac{1}{1-\gamma}\mathbb{E}_{o\sim d_\rho^\pi}\mathbb{E}_{a\sim\pi(\cdot|o)}\left[f(o,a)\right] \tag{Eq.19}
$$

其中折扣观测访问频率定义为：

$$
d_\rho^\pi(o) := (1-\gamma)\mathbb{E}_{s_1\sim\rho}\!\left[\sum_{h=0}^{+\infty}\gamma^h p(o_h=o\mid s_1;\pi)\right] \tag{Eq.20}
$$

**证明**（将积分顺序重排）：

$$
\text{LHS} = \int_\mathcal{S} ds_1\,\rho(s_1)\sum_{h=0}^{+\infty}\gamma^h\int_{\mathcal{O}\times\mathcal{A}} do_h\,da_h\; p(o_h\mid s_1)\,\pi_h(a_h\mid o_h)\,f(o_h,a_h)
$$

乘除 $(1-\gamma)$ 并识别定义 Eq.(20)：

$$
= \frac{1}{1-\gamma}\mathbb{E}_{s_1\sim\rho}\underbrace{(1-\gamma)\sum_{h=0}^{+\infty}\gamma^h\int_\mathcal{O} do\; p(o_h=o\mid s_1)}_{\text{定义}\; d_{s_1}^\pi(o)}\;\mathbb{E}_{a\sim\pi(\cdot|o)}f(o,a) = \frac{1}{1-\gamma}\mathbb{E}_{o\sim d_\rho^\pi}\mathbb{E}_{a\sim\pi(\cdot|o)}f(o,a)
$$

### 3.4 最终形式（平稳 POMDP，Eq.9 / Eq.21）

将 Eq.(19) 的引理作用于 $f = A^{\pi^\theta}\cdot\nabla_\theta\sum_k\ln\pi_\theta(a^{k+1}|a^k,o)$：

$$
\boxed{
\nabla_\theta J(\pi^\theta) = \frac{1}{1-\gamma}\;\mathbb{E}_{o\sim d_\rho^{\pi^\theta}}\;\mathbb{E}_{a^0,\ldots,a^K\sim\pi^\theta(\cdot|o)}\!\left[A^{\pi^\theta}(o,a^K)\;\nabla_\theta\!\sum_{k=0}^{K-1}\ln\pi_\theta(a^{k+1}\mid a^k,o)\right]
} \tag{Eq.9b / Eq.21}
$$

**直观理解**：
- $A^{\pi^\theta}(o, a^K)$ 评价**最终执行动作**的好坏（Critic）
- $\nabla_\theta\sum_k\ln\pi_\theta(a^{k+1}|a^k,o)$ 是**整条去噪轨迹的对数概率梯度**（同时作用于 $\theta$ 和 $\theta'$）
- 这意味着整条去噪链被**统一地**当作一个"策略选择单元"来更新，奖励信号从最终动作回传到每一个去噪步

---

## 4. PPO 实现（Clipped Surrogate Loss）

基于 Eq.(9b) 的优势估计，结合 PPO 的重要性采样裁剪（Schulman 等 2017），最终的优化目标为：

$$
\nabla_{\bar\theta}\;\frac{1}{B}\sum_{i=1}^B\left[-\min\!\left(\frac{\pi_{\bar\theta}(\mathbf{a}_i|o_i)}{\pi_{\bar\theta_\text{old}}(\mathbf{a}_i|o_i)}\widehat{A}_i,\;\operatorname{clip}\!\left(\frac{\pi_{\bar\theta}(\mathbf{a}_i|o_i)}{\pi_{\bar\theta_\text{old}}(\mathbf{a}_i|o_i)},\,1-\epsilon,\,1+\epsilon\right)\widehat{A}_i\right) + \alpha\cdot\mathcal{R}(\mathbf{a}_i,o_i;\bar\theta,\bar\theta_\text{old})\right]
$$

其中：

$$
\frac{\pi_{\bar\theta}(\mathbf{a}|o)}{\pi_{\bar\theta_\text{old}}(\mathbf{a}|o)} = \exp\!\left[\sum_{k=0}^{K-1}\ln\pi_{\bar\theta}(a^{k+1}|a^k,o) - \ln\pi_{\bar\theta_\text{old}}(a^{k+1}|a^k,o)\right]
$$

每一个 $\ln\pi_{\bar\theta}(a^{k+1}|a^k,o)$ 由 Eq.(7) 的高斯 log 概率给出，**无需重新采样**（用旧轨迹 $\mathbf{a}_i$ 即可计算新策略下的概率）。优势估计 $\widehat{A}_i$ 由 GAE（Schulman 等 2018）计算。

---

## 5. 正则化

### 5.1 Wasserstein-2（$W_2$）正则化

约束微调策略与预训练策略的 $W_2$ 距离，使用可计算上界：

$$
\mathcal{R}_{W_2}(\theta,\theta_\text{old}) = \mathbb{E}_o\,\mathbb{E}_{a\sim\pi_\theta(\cdot|o),\,a_\text{old}\sim\pi_{\theta_\text{old}}(\cdot|o)}\!\left[\frac{1}{2}\|a - a_\text{old}\|_2^2\right] \geq \mathbb{E}_o\!\left[W_2^2(\pi_{\theta_\text{old}}(\cdot|o),\pi_\theta(\cdot|o))\right]
$$

实践中：从同一初始噪声 $a^0 \sim \mathcal{N}(0,I)$ 出发分别采样 $a$（新策略）和 $a_\text{old}$（旧策略），不注入噪声，减小随机性影响。

### 5.2 熵正则化

对 Markov 过程参数化策略，采用**负块熵率（negative per-symbol entropy rate）**：

$$
\mathcal{R}_\mathbf{h}(\bar\theta) = -\frac{1}{K+1}\mathbb{E}\!\left[\mathbf{h}(a^0,\ldots,a^K\mid o;\bar\theta)\right]
= -\frac{1}{K+1}\mathbb{E}\!\left[\mathbf{h}(\mathcal{N}(0,I)) + \sum_{k=0}^{K-1}\mathbf{h}\!\left(\mathcal{N}(a^k+v_\theta\Delta t_k,\,\sigma_{\theta'}^2)\right)\right]
$$

利用高斯分布微分熵的封闭形式 $\mathbf{h}(\mathcal{N}(\mu,\sigma^2)) = \frac{d_A}{2}\ln(2\pi e\sigma^2)$，每项可显式计算：

$$
\mathbf{h}\!\left(\mathcal{N}(a^k+v_\theta\Delta t_k,\,\sigma_{\theta'}^2)\right) = \frac{d_A}{2}\ln(2\pi e) + d_A\ln\sigma_{\theta'}(t_k,a^k,o)
$$

最小化 $\mathcal{R}_\mathbf{h}$ ⟺ 最大化熵 ⟺ 鼓励噪声网络输出**更大的** $\sigma_{\theta'}$，从而促进探索。

---

## 6. 训练-推理非对称性：噪声网络的生命周期

$$
\underbrace{\text{预训练}}_{\pi_{bc}:\;v_\theta} \xrightarrow{\text{加噪声网络 }\sigma_{\theta'}} \underbrace{\text{RL 微调}}_{\pi_{rl}^{\bar\theta}:\;v_\theta+\sigma_{\theta'}} \xrightarrow{\text{丢弃 }\sigma_{\theta'}} \underbrace{\text{推理}}_{\pi_\text{deploy}:\;v_\theta\text{（新 }\theta\text{）}}
$$

- **微调中**：$\sigma_{\theta'}$ 和 $v_\theta$ 联合训练，由同一策略梯度损失驱动
- **推理时**：丢弃 $\sigma_{\theta'}$，策略退回到确定性 ODE（依然是 Rectified Flow / Shortcut Model），保持高效推理
- **参数量**：$\sigma_{\theta'}$ 仅为预训练策略的一小部分

---

## 7. 与 DPPO 的数学对比

| 维度       | [[wiki/concepts/DPPO]] | ReinFlow                           |
| -------- | ---------------------- | ---------------------------------- |
| 基础策略类    | DDPM（随机 SDE）           | Rectified Flow / Shortcut（确定性 ODE） |
| Log 概率来源 | DDPM 去噪链固有高斯性          | 注入噪声后 Markov 链封闭形式                 |
| 去噪步数     | 5步 DDIM                | **最少 1 步**                         |
| 探索机制     | DDPM 内置随机性             | $\sigma_{\theta'}$ 可学习探索强度         |
| 理论依据     | 双层 MDP + IS 比          | **Markov Process PG 定理（Thm 4.1）**  |
| 推理开销     | 中（DDIM 加速）             | **低（Shortcut 1步）**                 |

---

## 8. 全链路执行图

```
[预训练 Flow 策略 π_bc]
     ↓ v_θ (固定架构，更新参数)
[环境交互收集轨迹]
  a⁰ ~ N(0,I)
  a^(k+1) = a^k + v_θ(t_k,a^k,o)·Δt_k + σ_θ'(t_k,a^k,o)·ε   ← Eq.6
  a = a^K → 执行，得 reward r
     ↓
[计算 log prob (Eq.7)] → 联合 log π(a⁰,...,aᴷ|o)
     ↓
[GAE 计算优势 Â(o, aᴷ)]
     ↓
[PPO Clipped Surrogate Loss + 正则化 R]
     ↓ 梯度同时更新 θ 和 θ'
[新策略 π_rl = π^θ̄]
     ↓（收敛后）
[丢弃 σ_θ'] → 部署 v_θ（1步 Shortcut 推理）
```

---

## 相关页面

- 原论文摘要：[[wiki/sources/2026-04-18 ReinFlow]]
- 对比分析：[[RL 微调表达性策略方法对比]]
- 基础概念：[[wiki/concepts/Flow Matching]]、[[wiki/concepts/DPPO]]
- 方法对比中的并列工作：[[wiki/sources/2026-04-19 DPPO]]、[[wiki/sources/2026-04-19 Flow Q-Learning]]
