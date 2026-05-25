---
type: source
tags: [VLA-RL, Flow-Matching, FPO, π0, LIBERO, ALOHA, Latent-RL]
sources: [raw/sources/papers/VLA+RL/Lyu 等 - 2025 - Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models/Lyu 等 - 2025 - Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models.md]
created: 2026-05-25
updated: 2026-05-25
---

# π0-FPO: Reinforcement Fine-Tuning of Flow-Matching Policies for VLA (Lyu 2025)

> Mingyang Lyu, Yinqian Sun, et al. Chinese Academy of Sciences / UCAS.  
> **Note — FPO naming collision:** This paper independently names its algorithm "FPO (Flow Policy Optimization)". Two other papers use the same term: McAllister 2025 (source #29, original FPO) and Yi 2026 (source #43, FPO++). All three use the same core idea (CFM loss ratio as IS ratio), but are distinct works. In this wiki the disambiguation is: #29 = McAllister-FPO, #43 = FPO++ (Yi), #45 = π0-FPO (Lyu).

[[raw/sources/papers/VLA+RL/Lyu 等 - 2025 - Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models/Lyu 等 - 2025 - Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models.md]]

---

## TL;DR

π0-FPO enables **online RL fine-tuning of the π0 VLA model** by constructing a likelihood-free policy ratio from per-sample CFM loss changes. Beyond the basic CFM ratio, four additional components stabilize training: (1) structure-aware credit assignment in the latent space, (2) clipped PPO surrogate, (3) multi-step latent (Euler) exploration, and (4) a Q-ensemble critic. Achieves **87.2% avg on LIBERO**, **65.3% on LIBERO-Long**, and **>1.5× ALOHA-sim baseline** under sparse rewards.

![[raw/sources/papers/VLA+RL/Lyu 等 - 2025 - Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models/images/2eaa9685355951fd569b4d65509c376dec7b011237da5c71f4a1036c57b53284.jpg]]
*Figure (task benchmarks): ALOHA bimanual manipulation (TransferCube) and LIBERO manipulation (pick & place) — the two benchmark suites on which π0-FPO is evaluated.*

---

## 核心问题

π0 的 action 由 **flow-matching expert**（以 PaliGemma LLM 骨干为条件）生成，exact policy ratio 需解 ODE + Jacobian 迹积分，$\mathcal{O}(d^3 T)$ 不可行。奖励加权监督学习（RWFM）不支持主动探索，无法发现 OOD 策略。目标：为 π0 设计可扩展的 online RL 后训练。

---

## 方法：FPO（Lyu 版本）

### Actor-Critic 架构

**Actor**：π0 flow-based expert，在 **latent action 空间**操作。FPO 冻结 π0 的 base policy decoder，只更新 latent actor $\pi_\theta$，其生成 latent $x_t \in \mathbb{R}^D$，由 frozen decoder 解码为 action $a_t$。

**Critic**：Q-ensemble（多个 critic）在 latent 空间提供 advantage 估计。

### 核心：Likelihood-Free IS Ratio

对存储的 $(s_t, x_t)$ 对，用 CFM loss 减少量构造 ratio：

$$
\rho_t^\theta = \exp\!\Big(\ell_\text{cfm}(x_t|s_t;\theta_\text{old}) - \ell_\text{cfm}(x_t|s_t;\theta)\Big)
$$

此 ratio 无需显式密度或 ODE 求解，结构上与策略生成过程一致。

### 四个稳定化组件

| 组件 | 作用 |
|---|---|
| **Structure-Aware Credit Assignment** | 以 CFM latent 空间中的每样本改进量作为信号，提升梯度效率 |
| **Clipped PPO Surrogate** | $\operatorname{clip}(\rho, 1\pm\varepsilon)$ 稳定 trust region |
| **Multi-step Latent (Euler) Exploration** | 在 latent 空间做 Euler 步扰动，生成时序相关的平滑 latent 序列，扩大探索多样性 |
| **Q-Ensemble Critic** | 多 critic 聚合提供鲁棒 value 估计，抑制 Q-overestimation |

### 训练循环

**Rollout phase** → 存储 $(s_t, x_t)$ + per-sample CFM loss 到 sliding-window buffer  
**Update phase** → 重算 current actor 的 CFM loss → 构造 clipped ratio → critic Q-ensemble 提供 advantage → 更新 actor + critics

---

## 实验结果

| Benchmark | Metric | π0-FPO | SOTA Baseline |
|---|---|---|---|
| LIBERO avg | Success rate | **87.2%** | 60.2% (π0-FAST) |
| LIBERO-Long | Success rate | **65.3%** | 53.7% (OpenVLA baseline) |
| ALOHA Transfer Cube (sim) | Success rate | **>1.5× baseline** | — |

**消融**：structure-aware credit + Q-ensemble 各自贡献显著提升；multi-step latent exploration 改善稀疏奖励下早期收敛。

---

## 与同路线方法对比

| 方法 | IS Ratio | 额外稳定化 | 目标 VLA |
|---|---|---|---|
| McAllister FPO (#29) | CFM loss ratio (avg) | 无 | 通用 flow 策略 |
| FPO++ Yi 2026 (#43) | CFM loss ratio (per-sample) + ASPO | ASPO | 运动控制策略 |
| **π0-FPO Lyu 2025 (#45)** | **CFM loss ratio (per-sample)** | **Latent explorer + Q-ensemble** | **π0 VLA** |
| ReinFlow (#4) | Noise injection Markov PG | 封闭 log-prob | Flow 策略 |

---

## 关联知识

- [[wiki/sources/rl-finetuning/2026-05-17 FPO Flow Matching Policy Gradients (McAllister 2025)]] — 同核心思想，McAllister 独立来源
- [[wiki/sources/rl-finetuning/2026-05-25 FPO++ (Yi 2026)]] — 同核心思想，Yi 独立来源
- [[wiki/concepts/benchmarks/LIBERO]] — 主评测基准
- [[wiki/concepts/generative-models/Flow Matching]] — π0 action expert 的技术基础
- [[wiki/analyses/RL微调生成模型的技术挑战与解决方案]] — B 路线（CFM loss ratio）背景

## Knowledge Gaps（仅本来源）

- **Multi-step Latent (Euler) Exploration**：在 latent 空间做 Euler 扰动产生时序相关探索；仅来源 45（π0-FPO 独有设计）
- **Q-Ensemble Critic for Flow-VLA RL**：多 critic 集成在 flow-matching VLA latent 空间的 Q 估计；仅来源 45
