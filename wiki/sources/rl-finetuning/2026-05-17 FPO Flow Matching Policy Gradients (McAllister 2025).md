---
type: source
tags: [RL-Finetuning, Flow Matching, Policy Gradient, PPO, On-Policy, Multimodal Actions, Continuous Control]
sources: [raw/sources/papers/VLA+RL/McAllister - 2025 - Flow Matching Policy Gradients (FPO)/McAllister - 2025 - Flow Matching Policy Gradients (FPO).md]
created: 2026-05-17
updated: 2026-05-23
---

[[raw/sources/papers/VLA+RL/McAllister - 2025 - Flow Matching Policy Gradients (FPO)/McAllister - 2025 - Flow Matching Policy Gradients (FPO).md]]

# Flow Matching Policy Gradients (FPO)

**arXiv**: 2507.21053  
**作者**: David McAllister, Songwei Ge, Brent Yi, Chung Min Kim, Ethan Weber, Hongsuk Choi, Haiwen Feng, Angjoo Kanazawa  
**发表日期**: 2025-07-28  
**HuggingFace**: https://hf.co/papers/2507.21053（1 upvote）  
**摄取日期**: 2026-05-17  
**摄取来源**: arXiv 下载（agent 自动检索）

---

## 一句话摘要

Flow Policy Optimization (FPO) 将流匹配纳入策略梯度框架：以 **条件流匹配 loss 的优势加权比** 作为代理目标，兼容 PPO-clip，无需显式 log-prob，且对训练/推理时的扩散/流积分器选择完全不可知。

![[raw/sources/papers/VLA+RL/McAllister - 2025 - Flow Matching Policy Gradients (FPO)/images/26d146de8cc9bda6698932b6f5d16588b7794d832daa2f0fe1b3e8893903b401.jpg]]
*Figure：双目标 Gridworld 任务中 FPO 学到的 flow 与目标动作分布随去噪步演化 — 展示了 FPO 保留扩散策略多模态结构的能力。*

---

## 核心背景与动机

### 问题：流匹配策略做在线 RL 的两大障碍

1. **Log-probability 不可得**：标准策略梯度（PPO、TRPO）需要 $\log \pi_\theta(a \mid s)$；流模型用 ODE 定义，天然不提供封闭解析 log-prob（不同于 DPPO 的 DDPM 近似方案）
2. **采样器绑定**：多数扩散 RL 方法（DPPO、ReinFlow）的训练过程与特定采样器（DDPM、DDIM、Euler）深度绑定，换采样器即需重新推导

### 关键洞察

条件流匹配（CFM）的训练目标本身是一个 **加权回归 loss**，其比值与重要性采样比（IS ratio）在结构上完全类似。可以直接基于此构造 PPO-style surrogate 而无需计算 log-prob。

---

## 方法：Flow Policy Optimization (FPO)

### 核心公式

FPO 定义代理目标为：

$$r_t(\theta) = \frac{\mathcal{L}_{\text{CFM}}(a_t, s_t; \theta_{\text{old}})}{\mathcal{L}_{\text{CFM}}(a_t, s_t; \theta)}$$

其中 $\mathcal{L}_{\text{CFM}}$ 为条件流匹配 loss（velocity field 的 MSE）。策略更新目标：

$$\mathcal{L}_{\text{FPO}} = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right]$$

即标准 PPO-clip，将 IS ratio 替换为 CFM loss ratio。

### 关键设计选择

| 特性          | FPO                | DPPO（对比）        |
| ----------- | ------------------ | --------------- |
| Log-prob 需求 | ❌ 不需要              | ✅ 需要 DDPM 近似    |
| 采样器绑定       | ❌ 无（agnostic）      | ✅ 绑定特定 schedule |
| 训练模式        | On-policy          | On-policy       |
| 策略参数化       | 流匹配（Flow Matching） | DDPM            |

### 训练流程

1. 用当前策略 $\theta_{\text{old}}$ 与环境交互收集轨迹
2. 计算 Advantage 估计 $\hat{A}_t$（GAE）
3. 用收集的数据对 $\theta$ 做若干轮 FPO 更新
4. 重复

---

## 实验结果

### 连续控制基准

- FPO 可以**从头（scratch）**训练流匹配策略，不需要 BC 预训练
- 流匹配策略能捕捉**多模态动作分布**，在 under-conditioned 设置下优于高斯策略
- 在 Gym-Locomotion / DMControl 类任务上与 DPPO 相当或更优

### 关键优势场景

- **多模态任务**：流模型比单峰高斯策略更能覆盖多种可行动作
- **推理灵活性**：训练完成后可切换任意采样器（1步、4步、DDIM 等）而无需重训

---

## 与已有方法的关系

- **[[wiki/concepts/rl/DPPO]]**：FPO 的直接对比基线；FPO 用 CFM loss ratio 替代 DPPO 的显式 log-prob 方案
- **[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]]**：同为 on-policy + 流策略方向，但 ReinFlow 注入噪声 Markov 化实现显式 log-prob
- **[[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]]**：FlowRL taxonomy 将 FPO 归为"on-policy + CFM loss ratio"路线
- **[[wiki/concepts/generative-models/Flow Matching]]**：方法基础

---

## 新概念追踪

**首次出现，追踪中**：
- **CFM Loss Ratio 作为 IS Ratio 替代**：以条件流匹配损失比近似重要性采样比，规避 log-prob 计算；仅本来源系统提出

---

## 关联页面

- [[wiki/concepts/generative-models/Flow Matching]] — 方法基础
- [[wiki/concepts/rl/DPPO]] — 主要对比基线
- [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] — 同为 on-policy 流策略 RL 方向
- [[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]] — taxonomy 框架，含 FPO 归类
