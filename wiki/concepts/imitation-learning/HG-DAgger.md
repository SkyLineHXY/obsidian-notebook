---
type: concept
tags: [Imitation Learning, Online Learning, Human Intervention, Real-World RL, DAgger]
sources: [π₀.₆ 论文 (Intelligence 等 2025), RLinf GitHub README]
created: 2026-04-19
updated: 2026-04-19
---

# HG-DAgger — Human-Gated Dataset Aggregation

HG-DAgger（Human-Gated DAgger）是 **DAgger** 的一个变体，核心思想是由**人类专家主动决定何时介入**——仅在策略即将失败时接管，而不是在所有状态下都提供演示，从而高效采集高质量纠错数据。

---

## 背景：DAgger 回顾

经典 DAgger（Dataset Aggregation，Ross et al. 2011）通过以下迭代改进 BC：

$$
\mathcal{D}_{n+1} = \mathcal{D}_n \cup \{(o_t,\, a_t^{\text{expert}}) : o_t \sim d_{\pi_n}\}
$$

- $d_{\pi_n}$：当前策略 $\pi_n$ 的状态访问分布
- 每轮将策略 rollout 中产生的**真实状态分布**下的专家标注加入数据集

**局限**：每个时间步都需要专家在场并提供动作标注，真实部署中代价极高。

---

## HG-DAgger 机制

HG-DAgger 引入**人类门控**：专家只在认为机器人将要犯错时才介入，否则让机器人自主执行：

```
机器人自主执行 → 人类监视 → 发现潜在失败？
                      ↓ 是          ↓ 否
              人类接管并提供动作      机器人继续
                     ↓
         记录 (观测, 专家动作) 加入数据集 D
```

**核心特性**：
- 采集效率高：只在**接近失败的状态**采集专家数据，价值信噪比高
- 对分布偏移更鲁棒：专家在策略真实遇到的状态下纠错
- 对真实机器人友好：不需要全程专家操控

---

## 在 RECAP 框架中的作用

在 π₀.₆ 的迭代改进循环中，HG-DAgger 是数据采集的核心环节：

$$
\text{VLA rollout (自主)} + \text{HG-DAgger (专家介入)} \rightarrow \text{混合数据集} \mathcal{D}
$$

采集的数据包含：
1. 成功轨迹（$I=1$，正向 advantage 条件）
2. 失败轨迹的专家纠错段（$I=1$，分布修正）
3. 部分失败片段（$I=0$，负向 advantage 条件，让策略学习"不该做什么"）

---

## 与相关方法的比较

| 方法            | 专家介入频率  | 分布偏移处理 | 在线/离线 |
| ------------- | ------- | ------ | ----- |
| BC            | 零（离线演示） | 弱      | 离线    |
| DAgger        | 每步（全程）  | 强      | 在线    |
| **HG-DAgger** | 按需（失败前） | 强      | 在线    |
| RLHF          | 偏好标注    | 中      | 离线/在线 |

---

## 与 HIL-SERL 的关系

[[wiki/entities/systems/HIL-SERL]] 的人类介入机制在**形式上**与 HG-DAgger 相同：仅在策略即将失败时接管。但 HIL-SERL 将纠错数据用于**强化学习（RL）**而非监督学习，使策略能够超越人类演示，在正时皮带装配等 HG-DAgger 基线仅得 2% 成功率的任务上达到 100%。

---

## 出现来源

- [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] — HG-DAgger 作为 RECAP 数据采集策略
- [[wiki/sources/frameworks/2026-04-19 RLinf]] — RLinf 支持 HG-DAgger 用于 Franka 真实机器人在线训练
- [[wiki/sources/rl-finetuning/2026-04-24 HIL-SERL]] — HIL-SERL 与 HG-DAgger 进行系统对比，揭示 RL 与 SL 利用纠错数据的本质差异
