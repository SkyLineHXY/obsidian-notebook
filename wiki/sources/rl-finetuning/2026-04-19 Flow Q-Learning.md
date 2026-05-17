---
type: source
tags: [Flow Matching, 离线强化学习, 策略蒸馏, ICML 2025, Policy Extraction]
sources: [raw/sources/papers/VLA+RL/Park 等 - 2025 - Flow Q-Learning.pdf]
created: 2026-04-19
updated: 2026-04-19
---

[[Park 等 - 2025 - Flow Q-Learning.pdf]]

# Flow Q-Learning (FQL)

> **论文**：ICML 2025 | arXiv:2502.02538v2
> **作者**：Seohong Park、Qiyang Li、Sergey Levine（UC Berkeley）
> **项目主页**：https://seohong.me/projects/fql/

---

## 研究问题

**离线 RL 的本质**是**约束优化**：在数据集的状态-动作分布内最大化回报。随着数据规模扩大，行为分布越来越**多模态**，需要**表达性策略**来精确刻画行为约束。

但是用 Flow Matching（或 Diffusion）参数化策略在离线 RL 中遇到**策略抽取（policy extraction）难题**：
- 生成模型是**迭代**的，没有简单的方式最大化价值函数
- 若对迭代过程 BPTT，训练不稳定且计算昂贵
- 已有替代方案（加权回归、reparameterized PG、拒绝采样）各有 scaling 障碍

---

## 核心贡献

### 算法：FQL
**主要思想**：把两个角色分开处理。
- 用**行为克隆（BC）**训练一个**表达性 Flow 策略** $\mu^\beta(s, z)$，只负责刻画数据分布
- 再单独训练一个**一步策略** $\mu(s, z): \mathcal{S} \times \mathbb{R}^d \to \mathcal{A}$，负责**最大化 Q 值**，同时**蒸馏**自 BC flow 策略

$$
\min_\mu \; \underbrace{\mathbb{E}\bigl[-Q(s, \mu(s, z))\bigr]}_{\text{Q 最大化}} + \alpha\, \underbrace{\mathbb{E}\bigl[\|\mu(s, z) - \mu^\beta(s, z)\|^2\bigr]}_{\text{蒸馏正则}}
$$

### 关键优势
| 性质 | 说明 |
|------|------|
| **无 BPTT** | 价值最大化只作用于一步策略，不穿透 Flow 的 ODE |
| **推理时单步** | 部署时直接用 $\mu(s, z)$，无需 ODE 积分 |
| **保留表达力** | 通过蒸馏继承 BC Flow 的多模态刻画 |
| **实现极简** | 仅需几行代码叠加在 behavior-regularized actor-critic 框架上 |

### 背景：Behavior-Regularized Actor-Critic
在 TD3+BC 等方法上扩展的通用离线 RL 框架：

$$
L_Q(\phi) = \mathbb{E}\bigl[(Q_\phi - r - \gamma Q_{\bar\phi})^2\bigr], \quad
L_\pi(\theta) = \mathbb{E}\bigl[-Q_\phi(s, a) + \alpha\,\mathcal{L}_{\mathrm{BC}}\bigr]
$$

传统实现用高斯策略，FQL 用 Flow Matching 替换 BC 那一项，显著提升多模态分布的捕捉能力。

---

## 实验结果

- **73 个任务**：涵盖 OGBench（Park 2025）与 D4RL（Fu 2020），状态与像素两种观测
- 相比高斯策略 + actor-critic 与扩散策略离线 RL 方法，**在多模态分布任务上显著领先**
- 支持直接**离线-到-在线（offline-to-online）微调**，优于现有 offline-to-online baselines
- 推理速度接近高斯策略，无需迭代

---

## 学术定位

| 维度      | DPPO / ReinFlow  | **FQL（本文）**             |
| ------- | ---------------- | ----------------------- |
| 训练范式    | 在线 RL 微调         | 离线 RL（可 offline→online） |
| 是否 BPTT | 是（但通过 MDP 展开稳定化） | 否                       |
| 推理开销    | 多步去噪             | 单步                      |
| 策略表达    | 保留多步迭代 Flow      | 蒸馏到一步，表达性部分牺牲           |

FQL 代表了一条**与策略梯度正交**的技术路线：**把昂贵的表达性留给 BC，把便宜的价值优化留给一步策略**。

---

## 关联知识
- 核心概念页：[[wiki/concepts/generative-models/Flow Matching]]、[[wiki/concepts/rl/Offline 强化学习]]
- 对比分析：[[RL 微调表达性策略方法对比]]
- 相关工作：[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]]、[[wiki/sources/rl-finetuning/2026-04-19 DPPO]]
