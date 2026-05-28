---
type: concept
tags: [Offline RL, PPO, Diffusion Policy, OPE, Dataset Expansion, Real-World RL]
sources: [raw/sources/papers/VLA+RL/Lei 等 - 2026 - RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning/Lei 等 - 2026 - RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning.md]
created: 2026-05-28
updated: 2026-05-28
---

# Iterative Offline RL（迭代离线强化学习）

**Iterative Offline RL** 指一种**外层数据扩展 + 内层 OPE-gated 策略改进**的两级嵌套范式：在每轮外层迭代中，先用保守的 offline RL 在当前数据集 $\mathcal{D}_m$ 上改进策略，再用改进后的策略 rollout 收集新数据 $\mathcal{D}_{\text{new}}$ 合并，最后在扩展数据集 $\mathcal{D}_{m+1}$ 上做 IL 重训。该范式打破了纯 offline RL 受限于初始数据覆盖的根本瓶颈，同时通过 OPE（Offline Policy Evaluation）门控保证单调改善、避免 OOD 崩溃。

> **代表系统**：[[wiki/sources/rl-finetuning/2026-05-28 RL-100 Real-World RL on Diffusion Policy (Lei 2026)|RL-100]]（Lei 2026），在 8 个真实机器人任务上达到 1000/1000 成功率。
> **前驱方法**：BPPO（Behavior PPO）、Uni-O4——同样用 PPO clipped surrogate 在固定数据集上改进 IL 初始化策略，但未引入数据扩展飞轮。

---

## 1. 算法骨架（RL-100 Algorithm 1）

```
输入: 演示数据集 D₀, 外层迭代次数 M
初始化: π₀^IL ← ImitationLearning(D₀)

for m = 0 to M-1 do                              ← Outer Loop
    1. 训练 critic:    (Q_ψ, V_ψ) ← IQL(D_m)
    2. 训练转移模型:   T_θ(s' | s, a)            ← 给 AM-Q OPE 用
    3. Offline PPO 改进 (Inner Loop):
       for i = 0, 1, 2, ... do
           候选 π ← PPO_update(π_{m,i}, A_t^off)
           若 J^{AM-Q}(π) - J^{AM-Q}(π_{m,i}) ≥ δ:
               π_{m,i+1} := π                    ← 接受
           else:
               π_{m,i+1} := π_{m,i}              ← 拒绝
    4. Rollout:        D_new ← π_m rollout
    5. 数据合并:       D_{m+1} ← D_m ∪ D_new
    6. IL 重训:        π_{m+1}^IL ← IL(D_{m+1})
end for

7. 最终在线 RL:        π_final ← OnlineRL(π_{M-1}, V_ψ)
```

### 1.1 外层循环（Dataset-Expansion Round, index $m$）

| 步骤 | 操作 | 作用 |
|------|------|------|
| ① Critic 训练 | $(Q_\psi, V_\psi)\!\leftarrow\!\mathrm{IQL}(\mathcal{D}_m)$ | IQL expectile 回归给保守 advantage，防 OOD 高估 |
| ② Transition 模型 | $\hat{T}(s'\mid s,a)$ | OPE 门控的合成 rollout 引擎 |
| ③ Offline PPO 改进 | 内层 inner loop（见 §1.2） | 在 $\mathcal{D}_m$ 内做 OPE-gated 更新 |
| ④ Rollout | $\mathcal{D}_{\text{new}}=\{\tau\}\sim\pi_m$ | 真实环境收集高质量轨迹 |
| ⑤ 数据合并 | $\mathcal{D}_{m+1}=\mathcal{D}_m\cup\mathcal{D}_{\text{new}}$ | 数据分布单调拓展 |
| ⑥ IL 重训 | $\pi_{m+1}^{\mathrm{IL}}\leftarrow\mathrm{IL}(\mathcal{D}_{m+1})$ | 锚定到新分布，防 RL 漂移 |

**飞轮直觉**：更好的策略 → 更高质量 rollout → 数据覆盖更广 → critic 估计更准 / policy 学习更稳 → 又更好的策略。

### 1.2 内层循环（PPO Update, index $i$）

固定 $\mathcal{D}_m$，逐步推进行为策略 $\pi_{m,0}\to\pi_{m,1}\to\cdots$：

1. **候选生成**：在 $\mathcal{D}_m$ 上对当前 $\pi_{m,i}$ 做几轮 PPO clipped-surrogate 梯度更新得到 $\pi$。Advantage 来自 IQL critic：

   $$
   A_t^{\mathrm{off}} = Q_\psi(s_t, a_t) - V_\psi(s_t)
   $$

2. **OPE 门控（AM-Q Gate）**：用 Approximate Model-Q 在 $\hat{T}$ 上 rollout 候选策略评估累积 Q：

   $$
   \hat{J}^{\mathrm{AM-Q}}(\pi) = \mathbb{E}_{(s,a)\sim(\hat{T},\pi)}\!\left[\sum_{t=0}^{H-1} Q_\psi(s_t, a_t)\right]
   $$

3. **接受准则**（自适应阈值）：

   $$
   \hat{J}^{\mathrm{AM-Q}}(\pi) - \hat{J}^{\mathrm{AM-Q}}(\pi_{m,i}) \;\geq\; \delta,\quad \delta = 0.05\,|\hat{J}^{\mathrm{AM-Q}}(\pi_{m,i})|
   $$

   接受 → $\pi_{m,i+1}:=\pi$；拒绝 → $\pi_{m,i+1}:=\pi_{m,i}$。

4. **编码器策略**：内层全程共享冻结的 visual encoder $\phi$（IL 阶段训好），只更新 task-specific heads。

### 1.3 IL 重训为何关键（第 ⑥ 步）

RL-100 论文明确给出四条理由：

1. **Distribution shift**：新数据分布变了，IL 让策略跟随 $\mathcal{D}_{m+1}$。
2. **Stability anchor**：监督 IL 起 regularizer 作用，把策略锚定到高密度区，**防止纯 RL 探索造成的灾难性遗忘**。
3. **Multimodality preservation**：保留扩散策略表示多模态解的能力（纯 RL 优化易塌缩到单峰）。
4. **Distillation**：把人类演示 + RL 改进行为统一压进一个可部署策略。

---

## 2. 严谨数学推导

### 2.1 符号与目标

- 外层迭代索引 $m\in\{0,\dots,M-1\}$，对应数据集 $\mathcal{D}_m$。
- 内层迭代索引 $i$，行为策略 $\pi_{m,i}$（$\pi_{m,0}=\pi_m^{\mathrm{IL}}$）。
- 扩散策略每个环境步含 $K$ 步去噪，记去噪步索引 $k\in\{1,\dots,K\}$。
- 优化目标（最终）：

$$
\max_\pi\; J(\pi) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{H-1}\gamma^t R(s_t, a_t)\right]
$$

### 2.2 横跨去噪步的 Clipped PPO Surrogate

记环境级 advantage $A_t$，逐去噪步重要性比率

$$
r_k(\pi) = \frac{\pi(a^{\tau_{k-1}}\mid s^k)}{\pi_{m,i}(a^{\tau_{k-1}}\mid s^k)}.
$$

内层 PPO 目标：

$$
J_i(\pi) = \mathbb{E}_{s_t\sim\rho_{\pi_i},\,a_t\sim\pi_i}\!\left[\sum_{k=1}^{K}\min\!\Big(r_k(\pi)A_t,\;\mathrm{clip}(r_k(\pi),1-\epsilon,1+\epsilon)A_t\Big)\right]
\tag{Eq. 7}
$$

**关键设计**：所有 $K$ 个去噪步共享同一环境级 $A_t$ → 把稀疏的环境奖励信号"复制"到每个去噪步，缓解高维稀疏信用分配。

### 2.3 OPE-Monotonic Improvement 证明（直觉版）

设 $\pi_{m,i+1}$ 为接受后的策略。由接受准则：

$$
\hat{J}^{\mathrm{AM-Q}}(\pi_{m,i+1}) \;\geq\; \hat{J}^{\mathrm{AM-Q}}(\pi_{m,i}) + \delta
$$

若 $\hat{J}^{\mathrm{AM-Q}}$ 是真实 $J$ 的一致估计器（即 $\hat{T}\approx T$ 且 $Q_\psi\approx Q^\pi$），则在 $\hat{J}^{\mathrm{AM-Q}}$ 意义下序列单调上升。**实际中**$\hat{T}$ 和 $Q_\psi$ 都有误差，故引入 $\delta$ 的"安全余量"——只有候选策略**显著**优于当前才接受，过滤估计噪声。

### 2.4 与朴素 Offline PPO 的对比

记朴素方案为"无门控、$M=1$"：

| 性质 | 朴素 Offline PPO | Iterative Offline RL |
|------|------------------|----------------------|
| 数据支撑 | 仅 $\mathcal{D}_0$ | 单调扩展至 $\mathcal{D}_{M-1}$ |
| 接受策略 | 全部 | 仅 OPE 通过的更新 |
| 失败模式 | OOD-driven crash 常见 | OPE 过滤 + IL 锚定，飞轮稳步上升 |
| 在线衔接 | 切换易崩 | 同 PPO 目标 + 数据近 on-policy → 平滑过渡 |

---

## 3. 与其他 RL 范式的关系

| 范式 | 代表 | 与 Iterative Offline RL 的关系 |
|------|------|--------------------------------|
| 纯 Offline RL | IQL, CQL, [[wiki/concepts/rl/Offline 强化学习\|Offline RL]] | 提供 conservative critic 基础组件（IQL 在 RL-100 中即用） |
| 纯 Online RL | PPO, SAC | RL-100 在最后阶段衔接此方案（共享 surrogate） |
| Behavior-Cloned RL Init | RECAP, [[wiki/concepts/rl/AWR\|AWR]] | 同为"IL 初始化 + RL 改进"思路，但无数据扩展飞轮 |
| Hybrid IL+RL | [[wiki/entities/systems/HIL-SERL\|HIL-SERL]] | HIL-SERL 用 SAC + 人在环纠错；RL-100 用 PPO + OPE 门控 + 飞轮，无人在环 |
| Diffusion RL | [[wiki/concepts/rl/DPPO\|DPPO]] | DPPO 仅在线（仿真），RL-100 = 离线迭代 + 短在线 |

---

## 4. 实践要点

| 要点 | 建议 |
|------|------|
| Critic 选择 | 用 IQL / CQL 等保守变种，避免 OOD 高估传播到 PPO advantage |
| Transition 模型质量 | $\hat{T}$ 精度直接影响 AM-Q 可信度；建议短 horizon（H 较小）评估 |
| 阈值 $\delta$ | 5% 相对阈值是 RL-100 的经验值；越大越保守、收敛越慢 |
| 外层轮数 $M$ | RL-100 用 2–4 轮；视任务难度 |
| Rollout 量 | 每轮收集量需平衡部署成本与覆盖增益 |
| IL 重训权重 | RL-100 在 RL 阶段把 Recon/VIB 正则因子降 10×，IL 重训用原权重 |

---

## 5. 在知识库中的出现

| 来源 | 角色 |
|------|------|
| [[wiki/sources/rl-finetuning/2026-05-28 RL-100 Real-World RL on Diffusion Policy (Lei 2026)\|RL-100 (Lei 2026)]] | 主体：提出完整三阶段框架（IL → Iterative Offline → Online） |

**待入库前驱**（论文中提到但尚未单独建页）：
- **BPPO**（Behavior PPO）：用 PPO clipped surrogate 在离线数据上改进 BC 初始化策略
- **Uni-O4**：统一离线-在线 PPO 微调框架

---

## 6. 关联页面

- [[wiki/concepts/rl/Offline 强化学习]] — 上位概念
- [[wiki/concepts/rl/DPPO]] — 纯在线的扩散 PPO 微调
- [[wiki/concepts/rl/Representation Regularization in Visuomotor RL]] — RL-100 同篇配套的表征稳定器
- [[wiki/sources/rl-finetuning/2026-05-28 RL-100 Real-World RL on Diffusion Policy (Lei 2026)]] — 来源
- [[wiki/entities/systems/HIL-SERL]] — 关键对比系统
- [[wiki/analyses/RL微调生成模型的技术挑战与解决方案]] — 方法背景
