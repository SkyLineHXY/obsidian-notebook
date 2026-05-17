---
type: concept
tags: [Offline RL, 强化学习, 约束优化, 机器人学习, Behavior Regularization]
sources: [raw/sources/papers/VLA+RL/Park 等 - 2025 - Flow Q-Learning.pdf, raw/sources/papers/VLA+RL/Intelligence 等 - 2025 - $π^*_0.6$ a VLA That Learns From Experience.pdf]
created: 2026-04-19
updated: 2026-04-19
---

# Offline 强化学习

**Offline 强化学习（Offline RL / Batch RL）** 从**已收集的固定数据集** $D = \{(s, a, r, s')\}$ 出发，学习策略而**不与环境交互**。它的核心挑战是**分布外动作（OOD action）的价值被高估**——这使得朴素 Q-learning 在离线场景下严重发散。

---

## 核心形式化

给定 MDP $M = (\mathcal{S}, \mathcal{A}, r, \gamma, p)$ 与数据集 $D$（由某未知 **behavior policy** $\pi_\beta$ 采集），目标是：

$$
\max_\theta\; \mathbb{E}_{\tau \sim p_{\pi_\theta}}\Bigl[\sum_{h=0}^{H} \gamma^h r(s_h, a_h)\Bigr]
$$

**关键约束**：策略必须在数据分布内——这是与 online RL 的本质差异。

---

## 主流方法流派

| 流派 | 代表 | 核心思路 |
|------|------|---------|
| **保守/悲观 Q** | CQL、IQL | 惩罚 OOD 动作的 Q 值 |
| **Behavior-regularized Actor-Critic** | TD3+BC、BRAC | 在 actor loss 上加 BC 正则 |
| **加权回归** | AWAC、AWR、DT | 用 advantage 作权重的监督回归 |
| **生成模型策略抽取** | Diffusion-QL、**[[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning\|FQL]]** | 用扩散 / Flow 建模行为分布 |
| **Advantage Conditioning** | Decision Transformer、**[[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP\|RECAP]]** | 把 advantage / return 作为输入条件 |

---

## Behavior-Regularized Actor-Critic（共同骨架）

许多当代方法可统一在以下损失下：

$$
L_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim D,\,a' \sim \pi_\theta}\bigl[\bigl(Q_\phi(s,a) - r - \gamma Q_{\bar\phi}(s', a')\bigr)^2\bigr]
$$

$$
L_\pi(\theta) = \mathbb{E}_{s \sim D,\,a \sim \pi_\theta}\bigl[-Q_\phi(s, a)\bigr] + \alpha\, \mathcal{L}_{\mathrm{BC}}(\pi_\theta,\,\pi_\beta)
$$

$\alpha$ 控制**价值最大化 vs 行为约束**的拉锯。数据行为越多模态，$\mathcal{L}_{\mathrm{BC}}$ 就越需要**表达性策略类**来刻画。

---

## 为什么需要表达性策略？

标准 Gaussian 策略无法表达多模态动作分布，会坍缩到均值。随着机器人数据规模扩大、任务多样性提高，**多模态性几乎是默认假设**。这就推动了：
- **Diffusion-QL / IDQL**：用扩散建模 $\pi_\beta$
- **FQL**：用 [[wiki/concepts/generative-models/Flow Matching|Flow Matching]] 建模 $\pi_\beta$，并**蒸馏**到单步策略避免 BPTT
- **RECAP**：用 **advantage conditioning** 端到端训练大型 VLA

---

## Offline → Online 微调

**Offline-to-Online RL** 先离线预训练，再用少量在线交互继续提升：
- [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning|FQL]] 原生支持这一流程
- [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP|RECAP]] 的**迭代离线 RL** 本质上也是 offline → online 的大模型版本（每轮部署后把新数据并入离线数据集）

---

## 在知识库中的出现

| 来源 | 角色 |
|------|------|
| [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]] | **典型方法**：用 Flow 建模行为 + 单步蒸馏做 Q 最大化 |
| [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] | **大模型范式**：advantage-conditioned offline RL 应用于 VLA |

---

## 常用基准
- **D4RL**（Fu 2020）：经典离线 RL 基准
- **OGBench**（Park 2025）：更新、更大、更多任务
- **真实机器人数据集**：Open X-Embodiment、LeRobotDataset

---

## 与其他 RL 范式的位置

```
               Online RL（与环境实时交互）
                     │
                     ▼
  ── On-policy PG ── DPPO / ReinFlow / PPO on VLA ─ 
                     │
  ── Off-policy ─── SAC, TD3, Q-learning ──
                     │
                     ▼
               Offline RL（纯数据集）
                     │
                     ├── 保守/悲观（CQL、IQL）
                     ├── Behavior-regularized（TD3+BC、FQL）
                     └── Advantage conditioning（DT、RECAP）
                     │
                     ▼
           Offline-to-Online 微调（FQL、RECAP）
```
