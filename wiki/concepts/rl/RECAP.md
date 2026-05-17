---
type: concept
tags: [Offline RL, Advantage Conditioning, VLA, Imitation Learning, Diffusion Policy, Flow Matching]
sources: [π₀.₆ 论文 (Intelligence 等 2025), RLinf GitHub README]
created: 2026-04-19
updated: 2026-04-19
---

# RECAP — RL with Experience and Corrections via Advantage-conditioned Policies

RECAP 是一种**迭代离线 RL 框架**，由 Physical Intelligence 在 π₀.₆ 工作中提出，核心创新是将**价值优势（Advantage）作为条件前缀 Token** 喂给 VLA，绕开策略梯度估计的方差问题。

---

## 核心思想

传统 RL 微调大型 VLA 的困难在于：
1. 策略梯度方差大，难以扩展到大参数量
2. Flow Matching / Diffusion 动作头不易直接计算 log-prob

RECAP 的解法：**不做梯度更新，而是改变条件输入**。

$$
\hat{\pi}(a \mid o) \propto \pi_{\mathrm{ref}}(a \mid o)\, p(I \mid A_{\pi_{\mathrm{ref}}}(o, a))
$$

其中 $p(I \mid A)$ 是 advantage $A$ 的单调递增函数，使得 $J(\hat{\pi}) \ge J(\pi_{\mathrm{ref}})$。

**物理含义**：advantage 经二值化处理后作为 prefix token 输入策略，策略通过条件化学习"哪种动作质量更高"，而非通过梯度直接更新。

---

## 严谨数学推导

### 符号定义

| 符号 | 含义 |
|------|------|
| $\pi_{\mathrm{ref}}(a \mid o)$ | 参考策略（预训练 VLA）|
| $\hat{\pi}(a \mid o)$ | 改进后策略 |
| $V_\phi(o)$ | 语言条件化价值函数 |
| $A_{\pi}(o, a)$ | 动作 $a$ 在状态 $o$ 下的优势 $= Q(o,a) - V(o)$ |
| $I$ | 改进信号（binary：$I=1$ 表示高质量动作）|

### 推导步骤

**目标**：在不显式做策略梯度的前提下，找到 $\hat{\pi}$ 使 $J(\hat{\pi}) \ge J(\pi_{\mathrm{ref}})$。

**Step 1 — 从贝叶斯角度出发**

将"改进的策略"视为参考策略 $\pi_{\mathrm{ref}}$ 用改进条件 $I$ 进行后验加权：

$$
\hat{\pi}(a \mid o) = \pi_{\mathrm{ref}}(a \mid o, I=1) \propto \pi_{\mathrm{ref}}(a \mid o)\, p(I=1 \mid a, o)
$$

*这是标准的贝叶斯公式：将先验 $\pi_{\mathrm{ref}}$ 和似然 $p(I \mid a,o)$ 相乘得到后验。*

**Step 2 — 建立 Advantage 与改进条件的联系**

令 $p(I=1 \mid a, o) \propto \exp\!\left(\beta\, A_\pi(o, a)\right)$，其中 $\beta > 0$。代入得：

$$
\hat{\pi}(a \mid o) \propto \pi_{\mathrm{ref}}(a \mid o)\, \exp\!\left(\beta\, A_{\pi_{\mathrm{ref}}}(o, a)\right)
$$

*这与 AWR（Advantage-Weighted Regression）形式一致，有严格的改进保证：Jensen 不等式保证 $J(\hat{\pi}) \ge J(\pi_{\mathrm{ref}})$。*

**Step 3 — 二值化（实践简化）**

原始形式需要对每个动作计算 $\exp(\beta A)$，代价高。RECAP 将其离散化为二值：

$$
I = \mathbf{1}\!\left[A_{\pi_{\mathrm{ref}}}(o, a) > 0\right]
$$

然后训练时将 $I \in \{0, 1\}$ 编码为一个 prefix token，条件化注入 VLA：

$$
\pi_\theta(a \mid o, I) \quad \text{（标准 Teacher-Forcing 训练）}
$$

*二值化损失少量最优性，但使训练稳定且与 flow matching / diffusion 动作头兼容。*

**Step 4 — 价值函数训练**

价值函数 $V_\phi(o)$ 从多任务离线数据中训练，用 TD-bootstrapping 或蒙特卡洛回报估计：

$$
\mathcal{L}(\phi) = \mathbb{E}_{(o_t, r_t) \sim \mathcal{D}}\!\left[(V_\phi(o_t) - \hat{R}_t)^2\right]
$$

---

## 三阶段训练管线（π₀.₆）

```
Pre-training（大规模 offline RL）
    ↓
Task Specialization（演示微调）
    ↓
Iterative On-Robot Improvement（rollout + HG-DAgger + 更新 V + 更新 π）
```

---

## 与相关方法的比较

| 维度 | RECAP | [[wiki/concepts/rl/DPPO]] | [[wiki/concepts/generative-models/Flow Matching\|FQL / ReinFlow]] |
|------|-------|-------|------|
| 学习信号 | Offline Advantage | Online PPO | Online / Offline Q-value |
| 动作头兼容 | Flow Matching ✅ | Diffusion ✅ | Flow Matching ✅ |
| 策略梯度 | 否（条件化）| 是 | 否（Q-reg）/ 是（PG）|
| 定位 | 大型 VLA 系统 | 通用策略算法研究 | 通用策略算法研究 |

---

## 出现来源

- [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] — 原始提出论文（π₀.₆，Physical Intelligence 2025）
- [[wiki/sources/frameworks/2026-04-19 RLinf]] — RLinf 将 RECAP 作为 offline RL 算法集成支持
