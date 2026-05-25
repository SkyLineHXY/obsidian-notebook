---
type: source
tags: [RL, ActionChunking, OfflineRL, QlearningTheory, OGBench, BootstrappingBias]
sources: [raw/sources/papers/RL(Reinforce Learning)/Li 等 - 2025 - Decoupled Q-Chunking/Li 等 - 2025 - Decoupled Q-Chunking.md]
created: 2026-05-25
updated: 2026-05-25
---

# Decoupled Q-Chunking (DQC)

**Authors**: Qiyang Li, Seohong Park, Sergey Levine (UC Berkeley)
**Code**: github.com/ColinQiyangLi/dqc

[[raw/sources/papers/RL(Reinforce Learning)/Li 等 - 2025 - Decoupled Q-Chunking/Li 等 - 2025 - Decoupled Q-Chunking.md]]

---

## TL;DR

标准 TD 学习在长视野稀疏奖励任务中受困于 bootstrapping bias（自举偏差）。Chunked critics（评估动作序列而非单步动作）可通过 n-step return 缓解此问题，但迫使策略预测完整动作块，学习难度高、开环执行次优。

**核心洞见**：将 critic 的动作块大小（$h$）与策略的动作块大小（$h_a$）**解耦**。训练一个"蒸馏部分 critic" $Q_\psi^P(s_t, a_{t:t+h_a})$，通过 optimistic regression 近似原始 chunked critic 对部分动作块的最优值，策略只需预测短动作块 $h_a \leq h$（极端情况 $h_a=1$），同时保留 critic chunking 的价值传播加速。

---

## 核心贡献

1. **Action Chunking Q-learning 的理论基础**：首次形式化分析 Q-learning 中 action chunking 的 value estimation bias 与 suboptimality gap，引入 open-loop consistency（OLC）条件。
2. **Strong OLC 下的最优性保证**：若数据 $\mathcal{D}$ 强 $\varepsilon_h$-open-loop consistent，则 chunked Q-learning 收敛到近最优策略，次优性上界 $O(\varepsilon_h H \bar{H})$（与数据质量无关）。
3. **DQC 算法**：
   - 训练大块 critic $Q_\phi(s_t, a_{t:t+h})$ 用全块 TD backup
   - 通过 expectile regression 蒸馏部分 critic $Q_\psi^P(s_t, a_{t:t+h_a})$，近似完整块的最优值
   - 策略用 best-of-N 采样从行为流策略中提取，打分用 $Q_\psi^P$
4. **与 n-step return 的对比**：当数据 $\delta_n$-sub-optimal 时，只要 $\delta_n > 3\varepsilon_h \bar{H}$，action chunking Q-learning 的策略值**可证明优于** n-step return backup（因后者的 off-policy bias 与数据质量挂钩）。

---

## 关键图示

![[raw/sources/papers/RL(Reinforce Learning)/Li 等 - 2025 - Decoupled Q-Chunking/images/c7c8ffc6eddb79b0e766cf855df50a954aca8f4693bbc426c8bc9e21cfd2bb04.jpg]]
*Figure 1 左：DQC 核心思想。大块 critic（$h$ 大）负责价值加速学习，小块策略（$h_a$ 小）负责闭环反应执行。通过 distilled partial critic 桥接两者。*

![[raw/sources/papers/RL(Reinforce Learning)/Li 等 - 2025 - Decoupled Q-Chunking/images/54e236554276a709044809432cc8fdc9f70156103a67c56755098c25e13ff0ab.jpg]]
*Figure 1 右：DQC 在 OGBench 最难 6 个环境上的聚合得分，超越所有基线（QC=25 → DQC=82）。*

---

## 严谨数学推导

### 符号定义

| 符号 | 含义 |
|------|------|
| $\mathcal{M} = (\mathcal{S}, \mathcal{A}, T, r, \rho, \gamma)$ | MDP |
| $a_{t:t+h} = (a_t, \ldots, a_{t+h-1})$ | 长度 $h$ 的动作块 |
| $R_{t:t+h} = \sum_{t'=t}^{t+h-1}\gamma^{t'-t}r(s_{t'}, a_{t'})$ | $h$-step 累积奖励 |
| $\bar{H} = 1/(1-\gamma)$ | 标准有效视野 |
| $\bar{\bar{H}} = 1/(1-\gamma^h)$ | $h$-step 有效视野 |
| $Q_\phi(s_t, a_{t:t+h})$ | Chunked critic（大块，长度 $h$） |
| $Q_\psi^P(s_t, a_{t:t+h_a})$ | Partial critic（小块，长度 $h_a \leq h$） |
| $V_\xi(s_t)$ | 隐式 value function |
| $\pi_\beta$ | 行为流策略（flow matching BC） |

### 1. Action Chunking TD Backup

Chunked critic 的 TD 损失（$h$-step return，无 off-policy 偏差条件下）：

$$L(\phi) = \mathbb{E}_{s_{t:t+h+1}, a_{t:t+h}}\!\left[\Bigl(Q_\phi(s_t, a_{t:t+h}) - R_{t:t+h} - \gamma^h \bar{Q}(s_{t+h}, a_{t+h:t+2h}^\star)\Bigr)^2\right]. \tag{5}$$

关键性质：当数据满足强 open-loop consistency 时，此 backup 的 off-policy bias 为 $O(\varepsilon_h H \bar{H})$，**与 n-step return 的 $\delta_n$-sub-optimality 无关**。

### 2. Open-Loop Consistency（OLC）

**定义（弱 $\varepsilon_h$-OLC）**：对所有 $s_t \in \text{supp}(P_\mathcal{D})$，

$$D_\text{TV}\!\bigl(P_\mathcal{D}^\circ(s_{t+h'}, a_{t+h'} | s_t) \;\big\|\; P_\mathcal{D}(s_{t+h'}, a_{t+h'} | s_t)\bigr) \leq \varepsilon_h,\quad \forall h' \in \{1,\ldots,h-1\}. \tag{8-9}$$

**定义（强 $\varepsilon_h$-OLC）**：对所有 $a_{t:t+h} \in \text{supp}$，还要求逐轨迹：

$$D_\text{TV}\!\bigl(T(s_{t+h'} | s_t, a_{t:t+h'}) \;\big\|\; P_\mathcal{D}(s_{t+h'} | s_t, a_{t:t+h})\bigr) \leq \varepsilon_h,\quad \forall h' \in \{1,\ldots,h\}. \tag{10}$$

直觉：$\varepsilon_h$ 衡量"在数据集中按相同动作块开环执行后，状态分布与数据集真实分布的总变差距离"。当环境状态转移接近确定性且数据行为规律时，$\varepsilon_h$ 小。

### 3. Value Estimation Bias（Theorem 1 & 2）

若 $\mathcal{D}$ 弱 $\varepsilon_h$-OLC，则 chunked critic 名义值 $\hat{V}_\text{ac}$ 与真值 $V_\text{ac}$ 的偏差：

$$\left|V_\text{ac}(s_t) - \hat{V}_\text{ac}(s_t)\right| \leq \frac{\gamma\varepsilon_h}{(1-\gamma)(1-(1-\varepsilon_h)\gamma^h)} \leq \varepsilon_h H \bar{H}. \tag{12}$$

此界紧（Theorem 2 构造达下界的 MDP）。

### 4. AC Q-Learning 最优性（Theorem 3 & 4）

若 $\mathcal{D}$ 和 $\mathcal{D}^\star$ 强 $\varepsilon_h$-OLC 且 $\text{supp}(P_\mathcal{D}) \supseteq \text{supp}(P_{\mathcal{D}^\star})$，则学到的 AC 策略：

$$V^\star(s_t) - V_\text{ac}^+(s_t) \leq \frac{\varepsilon_h\gamma}{1-\gamma}\left[\frac{2}{1-(1-2\varepsilon_h)\gamma^h} + \frac{1}{1-(1-\varepsilon_h)\gamma^h}\right] \leq 3\varepsilon_h H \bar{H}. \tag{21}$$

**注意**：此界与数据质量（子优度）无关，仅与 $\varepsilon_h$ 和视野有关。而 n-step return 的 value 误差与 $\delta_n$（数据子优度）直接成正比。

### 5. DQC：Partial Critic Distillation

策略优化目标（仅预测 $h_a$ 步块）：

$$\pi^\star(s_t) = a_{t:t+h_a}^\star \;:=\; \arg\max_{a_{t:t+h_a}} Q_\phi(s_t, [a_{t:t+h_a},\, a_{t+h_a:t+h}^\star]). \tag{32-33}$$

**Partial critic 蒸馏**（近似上式中对 $a_{t+h_a:t+h}$ 的最大化）：

$$Q_\psi^P(s_t, a_{t:t+h_a}) \approx Q_\phi(s_t, [a_{t:t+h_a},\, a_{t+h_a:t+h}^\star]). \tag{34}$$

用 expectile regression（optimistic）蒸馏训练：

$$L(\psi) := f_\text{expectile}^{\kappa_d}\!\bigl(\bar{Q}_\phi(s_t, a_{t:t+h}) - Q_\psi^P(s_t, a_{t:t+h_a})\bigr). \tag{35}$$

$\kappa_d > 0.5$ 使 $Q_\psi^P$ 偏向高值——近似"对部分块延伸后可达最优值的最大 $\kappa$-expectile"。

**Value function**（用于 TD target，避免显式策略采样）：

$$L(\xi) = f_\text{quantile}^{\kappa_b}\!\bigl(\bar{Q}_\psi^P(s_t, a_{t:t+h_a}) - V_\xi(s_t)\bigr). \tag{38}$$

**Best-of-N 策略提取**（测试时）：

$$a_{t:t+h_a}^\star \leftarrow \arg\max_{\{a^i\}_{i=1}^N} Q_\psi^P(s_t, a_{t:t+h_a}),\quad a^i \sim \pi_\beta(\cdot|s_t). \tag{37}$$

行为策略 $\pi_\beta$ 用 flow matching BC 在离线数据集上训练，Best-of-N 提供 Q 值引导的隐式策略优化。

---

## DQC 算法（Algorithm 1）

```
Given: D, Q_φ(s_t, a_{t:t+h}), Q_ψ^P(s_t, a_{t:t+h_a}), V_ξ(s_t), π_β

Agent Update:
  Sample (s_{t:t+h+1}, a_{t:t+h}, r_{t:t+h}) ~ D
  Optimize Q_φ:  L(φ) = (Q_φ(s_t,a_{t:t+h}) - Σγ^k r_{t+k} - γ^h V̄_ξ(s_{t+h}))²
  Optimize Q_ψ^P: L(ψ) = f_expectile^κd(Q̄_φ(s_t,a_{t:t+h}) - Q_ψ^P(s_t,a_{t:t+h_a}))
  Optimize V_ξ:  L(ξ) = f_quantile^κb(Q̄_ψ^P(s_t,a_{t:t+h_a}) - V_ξ(s_t))

Policy Extraction:
  Sample N actions a^1,...,a^N ~ π_β(·|s_t)
  a* ← argmax Q_ψ^P(s_t, a_{t:t+h_a})
```

---

## 实验结果（OGBench）

[[wiki/concepts/benchmarks/OGBench]] 上 6 个最难环境（长视野目标条件离线 RL）：

| 方法 | 聚合得分 |
|------|---------|
| **DQC（本文）** | **82** |
| NS（n-step return） | 68 |
| SHARSA | 44 |
| QC（原始 Q-chunking，等块大小） | 25 |
| HIQL | 18 |

DQC 以显著优势超越所有基线，包括原始 Q-chunking（QC），证明解耦策略块大小的关键作用。

---

## 与相关工作的关系

| 方法 | Critic 类型 | 策略块 | off-policy bias | 策略学习难度 |
|------|------------|--------|----------------|------------|
| 1-step TD（IQL 等） | 单步 $Q(s,a)$ | $h_a=1$ | 高（bootstrapping bias） | 低 |
| n-step return | 单步 $Q$ | $h_a=1$ | 中（$\delta_n$ sub-optimal） | 低 |
| QC (Li 2025a) | 大块 $Q(s,a_{0:h})$ | $h_a=h$ | 低（OLC 条件下） | **高** |
| **DQC（本文）** | 大块 $Q$ + partial $Q^P$ | $h_a \leq h$ | 低 | **低** |

参见 [[wiki/concepts/imitation-learning/ACT]] 了解 action chunking 在模仿学习中的背景（Zhao 2023）。本文将 action chunking 正式化引入 Q-learning 的理论分析框架。

---

## Knowledge Gaps（待建概念页）

- **Action Chunking Q-learning**：本文首次提供理论基础（OLC 条件、value bias 上界）；与 [[wiki/concepts/imitation-learning/ACT]] 的 IL 视角互补；仅来源 48，待第二来源升级。
- **Open-Loop Consistency（OLC）**：DQC 理论核心，形式化动作开环回放的状态分布偏差；仅来源 48。
- **Bounded Optimality Variability（BOV）**：DQC 闭环执行最优性的替代条件；仅来源 48。
- **OGBench**：本文与 SAC Flow（来源 47）同时使用 → 已达 ≥2 阈值 → [[wiki/concepts/benchmarks/OGBench]]
