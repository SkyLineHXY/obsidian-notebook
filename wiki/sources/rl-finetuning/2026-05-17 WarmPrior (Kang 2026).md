---
type: source
tags: [RL-Finetuning, Flow Matching, Source Distribution, Behavior Cloning, Visuomotor Control, Robot Manipulation, Prior Design]
sources: [raw/assets/papers/VLA+RL/Kang - 2026 - WarmPrior Straightening Flow-Matching Policies with Temporal Priors.pdf]
created: 2026-05-17
updated: 2026-05-17
---

# WarmPrior: Straightening Flow-Matching Policies with Temporal Priors

**arXiv**: 2605.13959  
**作者**: Sinjae Kang, Chanyoung Kim, Kaixin Wang, Li Zhao, Kimin Lee  
**发表日期**: 2026-05-13  
**HuggingFace**: https://hf.co/papers/2605.13959（1 upvote）  
**摄取日期**: 2026-05-17  
**摄取来源**: arXiv 下载（agent 自动检索）

---

## 一句话摘要

将流匹配策略的**高斯源分布**替换为由近期动作历史构造的 **WarmPrior**（时序先验），使概率路径更直，在机器人操控 BC 任务和 prior-space RL 两条线上均提升性能与样本效率。

---

## 核心背景与动机

### 问题：Gaussian 源分布是一个未被充分探索的设计轴

当前生成策略（扩散/流匹配）的训练几乎统一使用标准高斯 $\mathcal{N}(0, I)$ 作为源分布。然而：

- **高斯源与动作目标之间的最优传输（OT）路径是弯曲的**，导致需要多步 ODE 积分才能得到准确样本
- **机器人动作在时序上高度相关**：相邻时刻的动作极为相近，而高斯源对此一无所知
- Rectified Flow 的研究（Liu 2022）已证明：OT 耦合产生更直的概率路径 → 更少积分步数 → 更低误差

### 关键洞察

无需计算真正的 OT 耦合（昂贵），可以用**近期动作历史**作为源分布的近似：历史动作与当前目标动作在分布上天然接近，从而产生更短、更直的流路径。

---

## 方法：WarmPrior

### 构造方式

$$q_{\text{WarmPrior}} = \mathcal{N}(\mu_{\text{hist}}, \sigma^2 I)$$

其中 $\mu_{\text{hist}}$ 从近期 $k$ 步动作历史计算（如均值或最后一步动作），$\sigma$ 为可调超参数。

训练时：从 WarmPrior 采样噪声 $x_0 \sim q_{\text{WarmPrior}}$ 而非 $\mathcal{N}(0,I)$，然后按标准流匹配流程训练 velocity field。

### 为什么有效

| 机制 | 效果 |
|---|---|
| 源与目标动作分布更接近 | ODE 路径更短更直 |
| 历史动作包含时序连续性先验 | 流学习的任务难度降低 |
| 路径更直 → 单步近似误差更小 | 推理步数可以减少 |

### 在 RL 中的应用

WarmPrior 同时也是 **prior-space RL** 的探索分布：用 WarmPrior 定义动作噪声的先验，RL 在此先验空间中探索 → 探索集中在高质量动作附近 → 样本效率提升。

---

## 实验结果

### 行为克隆（BC）

- 在多个机器人操控基准上，WarmPrior 一致性地提升任务成功率
- 效果等同于 Rectified Flow 的 OT 耦合，但无需计算 OT

### Prior-Space RL

- 用 WarmPrior 作为 RL 探索先验，同步提升样本效率和最终性能
- 消融实验确认：温度参数 $\sigma$ 控制探索宽度，存在最优值

---

## 与已有方法的关系

- **[[wiki/concepts/generative-models/Flow Matching]]**：方法基础；WarmPrior 是对流匹配源分布的替换
- **[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]]**：同为流策略 RL，ReinFlow 从目标函数角度（Markov 化）入手；WarmPrior 从源分布角度入手
- **[[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]]**：FQL 使用 one-step 流策略；WarmPrior 的直路径恰好也对 one-step 近似有利
- **[[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]]**：FlowRL taxonomy 中"源分布设计"维度的代表性工作

---

## 新概念追踪

**首次出现，追踪中**：
- **WarmPrior（时序动作先验）**：以近期动作历史为流匹配源分布，替代标准高斯；对 BC 和 RL 探索均有效；仅本来源
- **Prior-Space RL**：在流匹配先验空间定义探索分布的 RL 范式；仅本来源系统命名

---

## 关联页面

- [[wiki/concepts/generative-models/Flow Matching]] — 方法基础
- [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] — 同为流策略 RL 改进方向
- [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]] — one-step 流策略，受益于更直的路径
- [[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]] — taxonomy 框架，涵盖源分布设计轴
