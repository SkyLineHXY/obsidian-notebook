---
type: concept
tags: [Offline RL, Advantage Conditioning, VLA, Imitation Learning, Diffusion Policy, Flow Matching, CFGRL, AWR, Classifier-Free Guidance]
sources: [π₀.₆ 论文 (Intelligence 等 2025), RLinf GitHub README]
created: 2026-04-19
updated: 2026-05-17
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

| 符号                             | 含义                                    |
| ------------------------------ | ------------------------------------- |
| $\pi_{\mathrm{ref}}(a \mid o)$ | 参考策略（预训练 VLA）                         |
| $\hat{\pi}(a \mid o)$          | 改进后策略                                 |
| $V_\phi(o)$                    | 语言条件化价值函数                             |
| $A_{\pi}(o, a)$                | 动作 $a$ 在状态 $o$ 下的优势 $= Q(o,a) - V(o)$ |
| $I$                            | 改进信号（binary：$I=1$ 表示高质量动作）            |

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

## 条件 I 的核心作用：把 RL 改写成条件监督学习

### 为什么引入 I？

传统 RL（Actor-Critic / PPO）把 advantage 用作**梯度缩放因子**，直接参与反向传播：

$$
\nabla_\theta \mathcal{J} = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a \mid o) \cdot A(o,a)\right]
$$

[[wiki/concepts/rl/AWR]] 把 advantage 用作**样本权重** $\exp(A/\beta)$，做加权 BC——但对大模型不稳定，且不兼容 Flow Matching。

RECAP 的解法：**把 advantage 变成一个离散标签**。

$$
I_t = \mathbf{1}\!\left[A^{\pi_{\mathrm{ref}}}(o_t, a_t, \ell) > \epsilon_\ell\right]
$$

训练时在 prompt 里插入 `"Advantage: positive"` 或 `"Advantage: negative"`，策略学习**两个条件分布**：

| 条件 | 含义 | 训练时 | 推理时 |
|---|---|---|---|
| $I = \text{positive}$ | 该动作优于参考策略 | 从高质量动作中学习 | **强制设为此值** |
| $I = \text{negative}$ | 该动作劣于参考策略 | 从低质量动作学"不该做什么" | — |

推理时只需在 prompt 插入 `"Advantage: positive"`，模型自动生成改进策略下的动作。**负样本不被丢弃，而是作为反例主动学习**——这是 RECAP 超过 AWR 的关键原因。

### 信号传递路径对比

```
传统 Actor-Critic / PPO：
  advantage → 梯度缩放 → 策略权重更新（反向传播）

AWR：
  advantage → 样本权重 exp(A/β) → 加权 BC（加权反向传播）

RECAP：
  advantage → 二值化 → 条件 token I → 标准 CE 训练
```

### CFGRL：RECAP 的理论前驱

RECAP 引入 $I$ 的数学依据来自 Frans et al. CFGRL (2025) 的"鲜为人知的结果"：

正则化 RL 的最优策略不仅可以写成 AWR 的指数加权形式 $(\star)$，还可以等价地写为：

$$
\hat\pi(a \mid o) \propto \pi_{\mathrm{ref}}(a \mid o) \cdot p(I \mid A(o,a))^\beta \quad (\star\star)
$$

其中 $p(I \mid A)$ 是 advantage 的**任意单调递增函数**。这意味着：与其用连续权重 $\exp(A/\beta)$，不如引入一个**伯努利指示符 $I$**，其概率随 advantage 单调递增。两者同样保证策略改进，但后者天然兼容 Transformer 的条件生成范式（$I$ 只是一个 token）。

进一步，用贝叶斯公式翻转 $p(I \mid A)$：

$$
\hat\pi(a \mid o, \ell) \propto \pi_{\mathrm{ref}}(a \mid o, \ell)\left(\frac{\pi_{\mathrm{ref}}(a \mid I, o, \ell)}{\pi_{\mathrm{ref}}(a \mid o, \ell)}\right)^{\beta}
$$

当 $\beta=1$ 时，改进策略 $= \pi_{\mathrm{ref}}(a \mid I=1, o, \ell)$，**推理时强制 $I=\text{positive}$ 即可直接得到改进策略**，无需 CFG 插值，无需重新加权。详细推导见 [[wiki/analyses/π₀.₆ 与 RECAP 训练原理全景解析]]。

---

## 两种策略改进哲学

### PPO / Actor-Critic 的哲学：策略优化

> "我是一个策略 $\pi$，我需要被梯度推向更好的我。"

- 策略本身随训练改变
- 价值函数是指路灯塔，梯度是推动力
- 本质是**优化问题**：每一步梯度更新，策略向更好的方向移动

### RECAP 的哲学：条件建模

> "我是一个条件生成模型，我知道好动作和坏动作。你告诉我你想要好的，我就给你好的。"

- 策略不需要"被推向更好"
- 只需**学会区分好坏**，然后在推理时被条件激活
- 本质是**贝叶斯推断**：$\hat\pi(a \mid o) = \pi(a \mid I=\text{pos}, o)$

**工程含义**：策略梯度需要 log-prob 的梯度（对 Flow Matching 极难），条件生成只需标准 CE loss（Flow Matching 天然支持条件 token）。

---

## 与相关方法的比较

### vs. 传统 RL（Actor-Critic / PPO / AWR）

| 维度 | Actor-Critic / PPO | [[wiki/concepts/rl/AWR]] | **RECAP** |
|---|---|---|---|
| Advantage 的角色 | 梯度缩放因子 | 样本权重 $\exp(A/\beta)$ | 离散条件标签 $I$ |
| 更新方式 | 策略梯度 $\nabla_\theta \log\pi \cdot A$ | 加权行为克隆 | 标准交叉熵（CE） |
| 低 advantage 样本 | PPO clip；PPG 忽略 | 权重趋近 0，几乎丢弃 | 标记 $I=\text{neg}$，**主动学习** |
| Off-policy 处理 | 需要 importance sampling | 天然支持 | 天然支持 |
| Flow Matching 兼容 | ❌ 需要 log-prob 展开 | ❌ 破坏 CFM $\eta$-期望 | ✅ $I$ 只是文字 token |
| 自回归 token 兼容 | ❌ | ❌ | ✅ 条件 token 通吃 |
| 训练稳定性 | 需要 clip / KL penalty | 指数权重易爆炸 | 标准 CE 训练，稳定 |

### vs. 生成模型 RL 方法

| 维度 | RECAP | [[wiki/concepts/rl/DPPO]] | FQL / ReinFlow |
|------|-------|-------|------|
| 学习信号 | Offline Advantage 条件化 | Online PPO | Online / Offline Q-value |
| 动作头兼容 | Flow Matching ✅ | Diffusion ✅ | Flow Matching ✅ |
| 策略梯度 | 否（条件化）| 是 | 否（Q-reg）/ 是（PG）|
| 定位 | 大型 VLA 系统 | 通用策略算法研究 | 通用策略算法研究 |

详见 [[wiki/comparisons/VLA RL 微调方法对比]] 与 [[wiki/comparisons/RL 微调表达性策略方法对比]]。

---

## 出现来源

- [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] — 原始提出论文（π₀.₆，Physical Intelligence 2025）
- [[wiki/sources/frameworks/2026-04-19 RLinf]] — RLinf 将 RECAP 作为 offline RL 算法集成支持
