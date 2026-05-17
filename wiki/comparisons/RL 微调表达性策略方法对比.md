---
type: comparison
tags:
  - 强化学习
  - 扩散策略
  - Flow Matching
  - 策略微调
  - Offline RL
  - Online RL
  - PPO
  - Q-Learning
  - Taxonomy
sources:
  - raw/sources/papers/VLA+RL/Ren 等 - 2024 - Diffusion Policy Policy Optimization.pdf
  - raw/sources/papers/VLA+RL/Zhang 等 - 2026 - ReinFlow Fine-tuning Flow Matching Policy with Online Reinforcement Learning.pdf
  - raw/sources/papers/VLA+RL/Park 等 - 2025 - Flow Q-Learning.pdf
  - raw/sources/papers/VLA+RL/Luo 等 - 2025 - HIL-SERL.pdf
  - raw/assets/papers/VLA+RL/Gao - 2026 - FlowRL.pdf
  - raw/assets/papers/VLA+RL/McAllister - 2025 - FPO.pdf
  - raw/assets/papers/VLA+RL/Kang - 2026 - WarmPrior.pdf
  - raw/assets/papers/VLA+RL/Nguyen - 2025 - OFQL.pdf
  - raw/assets/papers/VLA+RL/Lee - 2026 - FAN.pdf
created: 2026-04-19
updated: 2026-05-17
---

# RL 微调表达性策略方法对比

> **覆盖范围**：策略骨干为**扩散模型或 Flow Matching**（无 LLM/VLM 骨干）的 RL 微调方法族，共 9 项工作：DPPO、ReinFlow、FQL、HIL-SERL、FlowRL（分类框架）、FPO、WarmPrior、OFQL、FAN。VLA 大模型 RL 后训练参见 [[wiki/comparisons/VLA RL 微调方法对比]]。

---

## 共同的技术难点

对于迭代生成策略 $\pi_\theta(a \mid o) = \int \prod_k p_\theta(a^{k-1} \mid a^k, o)\,\mathrm{d}a^{1:K}$：

1. **边缘似然 $\log\pi_\theta(a \mid o)$ 不可解析** → 标准策略梯度困难
2. **对去噪链 BPTT** → 训练不稳定、梯度方差爆炸、计算昂贵
3. **推理迭代多步** → 实时控制受限
4. **探索-利用权衡**：Flow 的确定性 ODE 无内置随机性；Diffusion 有噪声但微调可能破坏流形结构

---

## 方法全景：FlowRL Taxonomy 视角

根据 [[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]] 提出的统一分类体系，沿**优化目标 × 训练模式 × 去噪步数**三维展开：

| 方法                                                                                                  | 策略类                       | 训练范式                     | log-prob 方案                 | 推理步数      | 发表时间         |
| --------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------ | --------------------------- | --------- | ------------ |
| [[wiki/sources/rl-finetuning/2026-04-19 DPPO\|DPPO]]                                                | DDPM（扩散）                  | On-policy (PPO)          | 两层 MDP，每步高斯似然               | 多步 / DDIM | RSS 2025     |
| [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow\|ReinFlow]]                                        | Rectified Flow / Shortcut | On-policy (PPO)          | 噪声注入 → Markov → 封闭 log-prob | **≥ 1 步** | NeurIPS 2025 |
| [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning\|FQL]]                                      | Flow Matching             | Offline (Q + 蒸馏)         | 不需要（蒸馏到单步策略）                | **单步**    | ICML 2025    |
| [[wiki/sources/rl-finetuning/2026-04-24 HIL-SERL\|HIL-SERL]]                                        | 高斯（SAC）                   | Off-policy + HIL         | 高斯直接可算                      | N/A       | 2025         |
| [[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)\|FlowRL]]                                 | DDPM / Flow / Shortcut    | **Taxonomy + Benchmark** | —                           | —         | arXiv 2026   |
| [[wiki/sources/rl-finetuning/2026-05-17 FPO Flow Matching Policy Gradients (McAllister 2025)\|FPO]] | Flow Matching             | On-policy (PPO)          | CFM loss ratio ≈ IS ratio   | 多步（采样器无关） | arXiv 2025   |
| [[wiki/sources/rl-finetuning/2026-05-17 WarmPrior (Kang 2026)\|WarmPrior]]                          | Flow Matching             | BC / Prior-Space RL      | N/A（源分布设计，正交轴）              | 可减少步数     | arXiv 2026   |
| [[wiki/sources/rl-finetuning/2026-05-17 OFQL (Nguyen 2025)\|OFQL]]                                  | Flow Matching             | Offline (Q-learning)     | 不需要（avg velocity field 单步）  | **单步**    | arXiv 2025   |
| [[wiki/sources/rl-finetuning/2026-05-17 FAN (Lee 2026)\|FAN]]                                       | Flow Matching             | Offline (Q-learning)     | 不需要（单步 flow）                | **单步**    | arXiv 2026   |

---

## Log-prob 问题的三条解法路线

这是生成策略 RL 微调的**核心技术分歧点**。

### 路线 A：Markov 化去噪链（显式 log-prob）

DPPO 和 ReinFlow 的共同哲学：把多步去噪过程建模为 MDP，使每步转移具有可处理的高斯似然，从而启用 PPO。

$$\text{DPPO: } \log\pi(a^{k-1} \mid a^k, o) = \log\mathcal{N}(a^{k-1};\, \mu_\theta(a^k, o, k),\, \sigma_k^2 I)$$

$$\text{ReinFlow: } \log\pi(\mathcal{A} \mid o) = \log\mathcal{N}(0,I) + \sum_{k=0}^{K-1} \log\mathcal{N}\!\left(a_{k+1};\, a_k + v_\theta\Delta t_k,\, \sigma_{\theta'}^2\right)$$

区别：DPPO 在 DDPM 自带的高斯步中找到似然；ReinFlow 在 ODE 上**注入可学习噪声** $\sigma_{\theta'}$ 创造似然——推理时噪声网络丢弃，不影响 ODE 结果。

### 路线 B：CFM loss ratio 替代 IS ratio（FPO）

FPO 的洞察：条件流匹配训练目标（velocity MSE）与 PPO 中的重要性采样比在结构上类似。直接用 CFM loss 比构造 surrogate：

$$r_t(\theta) = \frac{\mathcal{L}_{\text{CFM}}(a_t, s_t;\, \theta_{\text{old}})}{\mathcal{L}_{\text{CFM}}(a_t, s_t;\, \theta)}, \quad \mathcal{L}_{\text{FPO}} = \mathbb{E}_t\!\left[\min\bigl(r_t \hat{A}_t,\; \text{clip}(r_t, 1\pm\varepsilon)\hat{A}_t\bigr)\right]$$

无需 log-prob，也无需绑定特定采样器。代价：IS ratio 的近似误差，理论保证弱于 A 路线。

### 路线 C：完全回避 log-prob（离线 Q-learning 族）

FQL、OFQL、FAN 统一思路：不做策略梯度，直接用 Q-value 驱动策略改进。

- **FQL**：BC-Flow 负责多模态表达力；单步蒸馏策略 $\mu(s,z)$ 专门最大化 Q 值。两者解耦，log-prob 无需触及。
- **OFQL**：学习 marginal **average velocity field** $\bar{u}_\theta$，直接实现单步映射 $a = x_0 + \bar{u}_\theta(x_0)$，消除迭代 ODE 与递归梯度。
- **FAN**：单步 flow-anchored 策略 + noise-conditioned 分布式 critic $Q_\phi(s, a, \epsilon)$，单噪声样本替代多分位数集合，理论上有收敛性与性能界保证。

---

## 核心机制深度对比

### On-policy PG 方法群（DPPO · ReinFlow · FPO）

| 维度          | DPPO                         | ReinFlow                                     | FPO                         |
| ----------- | ---------------------------- | -------------------------------------------- | --------------------------- |
| 策略类         | DDPM                         | Rectified Flow / Shortcut                    | Flow Matching               |
| Log-prob 方案 | 两层 MDP 高斯因子                  | 噪声注入 Markov 化                                | CFM loss ratio              |
| 探索机制        | DDPM 内置噪声 + noise schedule 改 | 可学习噪声 $\sigma_{\theta'}$                     | 流匹配采样随机性                    |
| 采样器绑定       | ✅ 绑定 DDPM schedule           | ✅ 绑定 ODE 步数（最少 1 步）                          | ❌ 完全无关                      |
| 去噪步数        | 多步（DDIM 可压缩）                 | **最少 1 步**                                   | 多步（推理后可切任意积分器）              |
| 结构化探索       | ✅（沿训练数据流形）                   | ✅（可调噪声幅度）                                    | △（取决于采样器随机性）                |
| 理论保证        | PPO 通用保证                     | Markov Process PG 定理 4.1                     | IS ratio 近似，较弱              |
| 关键数字        | FurnitureBench 主场            | Robomimic +40.34%；-62.82% wall-clock vs DPPO | Gym-Locomotion 与 DPPO 相当或更优 |

### Offline Q-learning 方法群（FQL · OFQL · FAN）

| 维度 | FQL | OFQL | FAN |
|---|---|---|---|
| 单步实现方式 | BC-Flow 预训练 + Q 最大化蒸馏（两阶段） | Marginal average velocity field 端到端训练 | Flow-anchored one-step + noise-conditioned critic（端到端）|
| Critic 类型 | 标准 Q-function | 标准 Q-function | Noise-conditioned $Q_\phi(s, a, \epsilon)$ |
| Offline-to-online | ✅ 支持 | ❓ 未报告 | ❓ 未报告 |
| 理论保证 | 蒸馏近似，较弱 | N/A | ✅ 收敛性 + 性能界证明 |
| 关键数字 | D4RL/OGBench 73 任务 SOTA | D4RL 超越 DQL，训练时间大幅缩短 | AntMaze/FrankaKitchen/Adroit SOTA |

**三者演进关系**：FQL 是先驱，用蒸馏实现单步；OFQL 和 FAN 各自找到不同的端到端单步方案，无需两阶段训练。FAN 额外引入分布式 critic 并提供最强理论保证。

### WarmPrior：正交的源分布设计轴

WarmPrior 不属于上面两个群，而是提出了一个**所有 flow 方法通用的改进轴**：将标准高斯源分布替换为时序动作先验：

$$q_{\text{WarmPrior}} = \mathcal{N}(\mu_{\text{hist}},\, \sigma^2 I)$$

其中 $\mu_{\text{hist}}$ 从近期 $k$ 步动作历史计算。机制：
- 源与目标动作分布更近 → ODE 路径更短更直 → 积分步数可减少
- 在 RL 中作为探索先验 → 探索集中在高质量动作附近 → 样本效率提升

**可叠加到 ReinFlow、FQL、FAN 等任意 flow 方法上，是正交改进。**

### HIL-SERL：系统级真实机器人参照

HIL-SERL 使用传统高斯策略（SAC + RLPD），不属于生成策略 RL 族，但作为**真实机器人 RL 的系统级参照**纳入对比：
- 13 个真实任务全部 100% 成功率（BC 基线仅 49.7%）
- 核心：**人类在线纠错**（HIL），纠错数据同时写入 demo buffer 和 RL buffer
- 无法直接与生成策略方法做公平横向比较，但作为**工程上限基准**有参考价值

---

## 方法论光谱（更新版）

```
  在线 RL ────────────────────────────────────────── 离线 RL
      │                                                   │
  DPPO ──── ReinFlow ──── FPO           FQL ─── OFQL ─── FAN
  (DDPM)   (Flow,PG)  (Flow,ratio)   (2-stage) (avg-v)  (noise-Q)
                                                   ↑
                              WarmPrior（正交轴：源分布替换，可叠加于任意方法）
  HIL-SERL（传统 SAC + 人类介入，系统级参照基线）
  FlowRL（统一分类框架 + JAX 高吞吐量 Benchmark）
```

---

## 设计哲学的根本分歧

| 核心哲学                       | 代表方法           | 论据                                       | 代价                             |
| -------------------------- | -------------- | ---------------------------------------- | ------------------------------ |
| **Markov 化 → 显式 PG**       | DPPO, ReinFlow | 去噪链可展开为 MDP，每步高斯似然可写；实证 sim2real 成功      | 绑定特定 schedule；ReinFlow 需额外噪声网络 |
| **CFM loss 结构替代 IS ratio** | FPO            | 无需 log-prob，无需绑定采样器，推理后可自由切换积分器          | IS ratio 近似误差；理论保证较弱           |
| **BC-表达力与 Q-优化解耦**         | FQL, OFQL, FAN | 避开昂贵 BPTT；单步推理高效；离线数据可直接利用               | 缺少在线探索；受限于离线数据集质量              |
| **时序先验优化 OT 耦合**           | WarmPrior      | Rectified Flow 理论：更直路径 → 更少步数；历史动作天然接近目标 | 需维护动作历史；超参数 $\sigma$ 敏感        |

---

## 定量基准对照

| 基准                   | DPPO        | ReinFlow                    | FQL        | FPO          | WarmPrior  | OFQL   | FAN      | HIL-SERL         |
| -------------------- | ----------- | --------------------------- | ---------- | ------------ | ---------- | ------ | -------- | ---------------- |
| FurnitureBench 长时序装配 | ⭐⭐⭐         | —                           | —          | —            | —          | —      | —        | —                |
| OpenAI Gym 运动控制      | —           | Rectified Flow **+135.36%** | —          | ✓            | ✓（BC+RL提升） | —      | —        | —                |
| Robomimic（视觉）        | ✓           | Shortcut **+40.34%**        | ✓          | —            | —          | —      | —        | —                |
| D4RL / OGBench       | —           | —                           | 73 任务 SOTA | —            | —          | 超越 DQL | **SOTA** | —                |
| IsaacLab 操控（GPU 仿真）  | —           | —                           | —          | —            | ✓          | —      | ✓        | —                |
| DMControl            | —           | —                           | —          | ✓（与 DPPO 相当） | —          | —      | —        | —                |
| 真实机器人操控              | sim→real 装配 | —                           | —          | —            | —          | —      | —        | **100% × 13 任务** |

> FlowRL (Gao 2026) 在 Gym-Locomotion / DMControl / IsaacLab 三套基准上提供跨方法的系统化横向比较，是目前最权威的公平对比参考。

---

## ReinFlow vs DPPO 直接对照

| 维度            | DPPO                          | ReinFlow                   |
| ------------- | ----------------------------- | -------------------------- |
| 策略类           | DDPM                          | Flow Matching              |
| 去噪步数下限        | ~10（DDIM 加速）                  | **1 步**                    |
| Wall-clock 时间 | 基准                            | **-62.82%**（全任务平均）         |
| 探索方式          | DDPM 内置噪声 + 修改 noise schedule | 可学习噪声网络 $\sigma_{\theta'}$ |
| 推理时额外开销       | 无（噪声 schedule 不变）             | 无（噪声网络微调后丢弃）               |

---

## 实用选型建议

| 场景                                | 推荐方法                 | 理由                                |
| --------------------------------- | -------------------- | --------------------------------- |
| 有仿真器，已有 Diffusion Policy baseline | **DPPO**             | 成熟事实基线，结构化探索                      |
| 有仿真器，Flow Matching 策略，追求快速推理      | **ReinFlow**         | 可低至 1 步推理，wall-clock -62%         |
| 有仿真器，Flow 策略，不想绑定特定采样器            | **FPO**              | 采样器无关，训练后可自由切换推理积分器               |
| 只有静态数据集，高维多模态分布                   | **FQL**              | offline-to-online 支持良好，73 任务 SOTA |
| 静态数据集，需要极快推理，不想两阶段训练              | **OFQL / FAN**       | 端到端单步，无需蒸馏                        |
| 需要 offline RL 的理论保证               | **FAN**              | 有收敛性 + 性能界证明                      |
| 想给任何 flow 方法免费加速                  | **WarmPrior（叠加）**    | 正交改进，与任意 flow 方法兼容                |
| 真实机器人，需要人类辅助 bootstrap            | **HIL-SERL**         | 1–2.5 小时收敛，13 任务 100%             |
| 研究/教学，需要公平跨方法比较                   | **FlowRL benchmark** | 统一框架 + JAX 高吞吐实现                  |

---

## 相关页面

- **VLA 骨干 RL 后训练**：[[wiki/comparisons/VLA RL 微调方法对比]]
- 策略参数化：[[wiki/concepts/generative-models/Diffusion Policy]]，[[wiki/concepts/generative-models/Flow Matching]]
- RL 范式：[[wiki/concepts/rl/Offline 强化学习]]，[[wiki/concepts/rl/DPPO]]
- 源页面：[[wiki/sources/rl-finetuning/2026-04-19 DPPO]]，[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]]，[[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]]，[[wiki/sources/rl-finetuning/2026-04-24 HIL-SERL]]，[[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]]，[[wiki/sources/rl-finetuning/2026-05-17 FPO Flow Matching Policy Gradients (McAllister 2025)]]，[[wiki/sources/rl-finetuning/2026-05-17 WarmPrior (Kang 2026)]]，[[wiki/sources/rl-finetuning/2026-05-17 OFQL (Nguyen 2025)]]，[[wiki/sources/rl-finetuning/2026-05-17 FAN (Lee 2026)]]
