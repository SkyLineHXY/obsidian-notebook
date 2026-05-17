---
type: concept
tags: [生成模型, Flow Matching, Rectified Flow, ODE, 策略表达, 机器人学习]
sources: [raw/sources/papers/VLA+RL/Zhang 等 - 2026 - ReinFlow Fine-tuning Flow Matching Policy with Online Reinforcement Learning_1.pdf, raw/sources/papers/VLA+RL/Park 等 - 2025 - Flow Q-Learning.pdf, raw/sources/papers/VLA+RL/Intelligence 等 - 2025 - $π^*_0.6$ a VLA That Learns From Experience.pdf, raw/sources/papers/Generative Model/Lipman 等 - 2023 - Flow Matching for Generative Modeling.pdf]
created: 2026-04-19
updated: 2026-04-21
---


# Flow Matching

**Flow Matching**（Lipman 等 2023；Liu 等 2023；Albergo & Vanden-Eijnden 2023）是一类**训练更简洁、推理更快**的生成模型，是**[[wiki/concepts/generative-models/Diffusion Policy|扩散模型]]的替代方案**。在机器人学习中，Flow Matching 正在取代 DDPM 成为**表达性策略的新默认参数化**。

---

## 核心数学

给定数据分布 $p(x) \in \Delta(\mathbb{R}^d)$，拟合一个时间相关速度场 $v_\theta(t, x): [0, 1] \times \mathbb{R}^d \to \mathbb{R}^d$，其对应的**流**由 ODE 定义：

$$
\frac{d}{dt} \phi(t, x) = v_\theta(\phi(t, x)), \qquad \phi(0, x) \sim \mathcal{N}(0, I_d), \;\phi(1, x) \sim p(x)
$$

**线性路径 + 均匀时间采样**（Rectified Flow）的目标函数：

$$
\min_\theta\; \mathbb{E}_{x_0 \sim \mathcal{N}(0, I),\, x_1 \sim p(x),\, t \sim \mathrm{Unif}[0,1]} \bigl\|v_\theta(t, x_t) - (x_1 - x_0)\bigr\|_2^2, \qquad x_t = (1-t)x_0 + tx_1
$$

**推理**：数值求解 ODE（最简 Euler 即可）从噪声到数据。

---

## 与 Diffusion 的对比

| 维度   | DDPM / Diffusion        | Flow Matching                 |
| ---- | ----------------------- | ----------------------------- |
| 数学基础 | 随机微分方程（SDE）             | 常微分方程（ODE）                    |
| 训练路径 | 高斯前向扩散                  | 直线插值（Rectified）               |
| 推理步数 | 通常 100+，DDIM 可降至 10–50  | **少至 1–4 步**（shortcut models） |
| 训练难度 | 需 noise schedule、SNR 权重 | 损失更简单、参数更少                    |
| 生成质量 | 成熟                      | 与 SDE 相当或更好（Esser 等 2024）     |

---

## 作为策略参数化的优势与挑战

### 优势
- **推理快**：直线 ODE，机器人实时控制更友好
- **表达力与扩散相当**：能刻画多模态动作分布
- **训练稳定**：损失结构类似 Flow Matching，无需 noise schedule 调参

### 挑战：**RL 微调困难**
1. **Log 概率难以计算**：确定性 ODE 本身不是概率分布，对比 DDPM 的离散去噪步有天然高斯似然
2. **缺乏内置探索**：路径确定，无噪声

各家方法针对这一挑战的解法（详见 [[RL 微调表达性策略方法对比]]）：
- **[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow|ReinFlow]]**：注入可学习噪声，把 ODE 离散化为 Markov 过程，从而得到 log-prob 的封闭形式
- **[[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning|Flow Q-Learning]]**：把表达性留给 BC Flow，单独训一个一步策略去最大化 Q
- **[[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP|RECAP（π₀.₆）]]**：advantage conditioning 替代策略梯度

---

## 在知识库中的出现

| 来源 | 角色 |
|------|------|
| [[wiki/sources/generative/2026-04-19 Flow Matching (Lipman 2023)]] | **原始论文**：Flow Matching 方法论奠基 (ICLR 2023) |
| [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] | **RL 微调**对象：Rectified Flow + Shortcut Models |
| [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]] | **离线 RL** 的行为先验 |
| [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] | **大型 VLA 的动作头**参数化 |

---

## 相关变种
- **Rectified Flow**（Liu 等 2023）：线性路径版本，绝大多数机器人工作采用
- **Shortcut Models**（2024）：一步 / 少步生成，ReinFlow 在该类策略上做微调
- **Conditional Flow Matching**（CFM）：条件生成扩展

---

## 完整数学推导

详见 → [[wiki/analyses/Flow Matching 完整数学推导]]

涵盖：符号定义 · ODE/流映射 · 连续性方程 · FM 损失不可计算的原因 · CFM 技巧 · **CFM = FM 梯度等价完整证明** · Rectified Flow 线性路径推导 · 训练与推理算法 · 与扩散模型的统一视角。

---

## 知识缺口
- **Shortcut Models 具体机制**（仅 ReinFlow 涉及，待第二篇来源出现后建页）
