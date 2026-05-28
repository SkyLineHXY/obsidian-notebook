---
type: source
tags: [Reinforcement Learning, Diffusion Policy, Real-World RL, PPO, Consistency Distillation, Robotic Manipulation]
sources: [raw/assets/papers/VLA+RL/Lei 等 - 2026 - RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning.pdf]
created: 2026-05-28
updated: 2026-05-28
---

# RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning

**arXiv**: 待确认
**作者**: Kun Lei†, Huanyu Li†, Dongjie Yu†, Zhenyu Wei†, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, Huazhe Xu\*
**机构**: Shanghai Qi Zhi Institute / SJTU / HKU / Tsinghua / UNC / CMU / CASIA
**项目页**: https://lei-kun.github.io/RL-100/
**摄取日期**: 2026-05-28
**摄取来源**: 用户添加 PDF + MinerU 转换

[[raw/assets/papers/VLA+RL/Lei 等 - 2026 - RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning.pdf]]
[[raw/sources/papers/VLA+RL/Lei 等 - 2026 - RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning/Lei 等 - 2026 - RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning.md]]

---

## 一句话摘要

RL-100 在扩散策略上统一 PPO surrogate 目标，通过三阶段（IL→迭代离线 RL→在线 RL）+ 一致性蒸馏，在 8 个真实机器人任务上实现 1000/1000 完美成功率，并完成商场连续 7 小时无故障部署。

---

## 核心背景与动机

### 问题：扩散策略 RL 微调的信用分配困难

扩散策略通过 K 步去噪生成动作，若仅在最终动作上施加 RL 奖励，梯度信号极度稀疏（"高维、多步、稀疏信用分配"问题）。同时，离线→在线切换时若优化目标不同，会出现性能骤降（performance crash）。

已有方法 BPPO 和 Uni-O4 用 PPO 裁剪 surrogate 在离线阶段改善模仿初始化策略，但未扩展到复杂真实机器人的扩散策略+分块动作设置。

### 关键洞察

将去噪过程建模为**两级 MDP**（环境 MDP + 去噪 MDP），在所有 K 去噪步上**共享同一环境级 Advantage $A_t$**，使每个去噪步都获得密集学习信号，同时用统一的 PPO clipped surrogate 在离线和在线阶段无缝切换。

---

## 方法：RL-100 三阶段框架

### 阶段 1：模仿学习（IL）预训练

扩散策略以噪声预测目标训练：

$$
\mathcal{L}_{\mathrm{IL}}(\theta) = \mathbb{E}_{(a^{\tau_0}, c_t)\sim\mathcal{D},\,\tau,\,\varepsilon}\left[\|\varepsilon - \varepsilon_\theta(a^\tau, \tau, c_t)\|_2^2\right] \tag{3}
$$

视觉编码器加入**重建正则化**（Chamfer distance）和**变分信息瓶颈（VIB）**稳定 RL 微调时的表征：

$$
\mathcal{L}_{\mathrm{recon}} = \beta_{\mathrm{recon}}\left(d_{\mathrm{Chamfer}}(\hat{o}, o) + \|\hat{q} - q\|_2^2\right), \quad \mathcal{L}_{\mathrm{KL}} = \beta_{\mathrm{KL}}\,\mathrm{KL}(\phi(z|o,s)\,\|\,\mathcal{N}(0,\mathbf{I})) \tag{4,5}
$$

### 阶段 2：迭代离线 RL（核心创新）

**两级 MDP 结构**：
- 外层 MDP：环境步 $t$，动作 $a_t$，奖励 $R_t$
- 内层 MDP：K 步去噪过程，每步生成 $a_t^{\tau_{k-1}}$

**统一 PPO 目标**（横跨所有去噪步 $k$，共享环境级 Advantage $A_t$）：

$$
J_i(\pi) = \mathbb{E}_{s_t\sim\rho_{\pi_i},\,a_t\sim\pi_i}\left[\sum_{k=1}^K \min\!\left(r_k(\pi)A_t,\; \mathrm{clip}(r_k(\pi), 1-\epsilon, 1+\epsilon)\,A_t\right)\right] \tag{7}
$$

其中 $r_k(\pi) = \pi(a^{\tau_{k-1}}\mid s^k)\,/\,\pi_{m,i}(a^{\tau_{k-1}}\mid s^k)$ 为逐去噪步的重要性比率。Advantage 由 IQL 的保守估计 $A_t^{\mathrm{off}} = Q_\psi(s_t,a_t) - V_\psi(s_t)$ 提供。

**OPE 门控（AM-Q gate）**：每次内循环更新前，用近似模型 Q 函数（AM-Q）评估候选策略：

$$
\hat{J}^{\mathrm{AM-Q}}(\pi) = \mathbb{E}_{(s,a)\sim(\hat{T},\pi)}\left[\sum_{t=0}^{H-1}Q_\psi(s_t,a_t)\right]
$$

只有当 $\hat{J}^{\mathrm{AM-Q}}(\pi) - \hat{J}^{\mathrm{AM-Q}}(\pi_{m,i}) \geq \delta$ 时才接受更新 → 保证单调改善，避免过拟合数据集外动作。

**迭代流程**（Algorithm 1）：每轮：训练 Critic → 离线 RL 优化 → 策略展开收集新数据 → 合并数据集 → IL 重训练。多轮后数据覆盖不断扩展，策略质量飞轮式提升。

### 阶段 3：短暂在线 RL

用相同 PPO 目标（在线 $r_k^{\mathrm{on}}$），少量真实 rollout（约 434 episodes/任务）消除残余失败模式。

### 一步一致性蒸馏（CM）

将 K 步 DDIM 策略蒸馏为一步一致性模型（Consistency Model），实现高频控制（无推理延迟）：

$$
\pi_m^{\mathrm{cm}} \leftarrow \mathrm{ConsistencyDistillation}(\pi_m^{\mathrm{ddim}})
$$

---

## 实验结果

**任务覆盖**（8 个真实机器人任务）：Dynamic Push-T、Agile Bowling、Pouring、Soft-towel Folding、Dynamic Unscrewing、Orange Juicing×2、Box Folding。跨 UR5 / Franka+LeapHand / xArm-Franka 三种机械臂。

**成功率（Table 1）**：

| 方法 | 均值成功率 |
|------|------|
| DP-2D（纯 IL）| 45.3% |
| DP3（纯 IL）| 67.8% |
| 迭代离线 RL | 91.8% |
| **RL-100 DDIM** | **100%（450/450）** |
| **RL-100 CM** | **100%（550/550）** |

核心亮点：
- Soft-towel Folding CM：250/250 连续成功
- Box Folding：DP-2D 12% → RL-100 100%（+88 points）
- 商场连续部署 7 小时无故障零样本迁移

**效率（vs 人类遥操）**：Dynamic Push-T 上 RL-100 DDIM 吞吐量 20 episodes/unit time，超越人类专家（17）和新手（13）。Box Folding 执行时间从 65.1s 降至 41.4s（1.57× 更快）。

**泛化鲁棒性**：
- 零样本迁移：平均 90%（表面摩擦变化、视觉干扰、OOD 初始位置）
- 少样本适应（1-3h）：平均 86.7%
- 抗外部扰动：平均 96%（持续 4s 反向力、拖拽、碰触）

---

## 局限性与未来工作

- 一步 CM 策略对噪声敏感（Orange Juicing Removal 因 IK 不连续性未评测 CM）
- 自动 reset 机制仍是瓶颈，需人工参与
- 当前仅在单任务级部署；跨任务/跨机身缩放有待验证
- 计划扩展至大规模多任务 VLA 后训练

---

## 与已有方法的关系

- **[[wiki/sources/rl-finetuning/2026-04-19 DPPO]]**：同为"PPO 微调扩散策略"方向，但 DPPO 在仿真中，RL-100 专注真实机器人部署级可靠性
- **[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]]**：Flow Matching 策略的 RL 微调，与 RL-100 扩散策略路线互补
- **[[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]]**：同类综述/分类框架，RL-100 可视为"完整真实机器人部署系统"的代表
- **[[wiki/entities/systems/HIL-SERL]]**：关键对比基线，RL-100 在扩散策略基础上增加了 OPE 门控和迭代离线阶段

---

## 新概念追踪

**首次出现，追踪中**：
- **两级 MDP（Denoising MDP + Environment MDP）**：将扩散去噪建模为嵌套 MDP，使 RL 信号在所有去噪步传播；仅来源 50
- **OPE 门控（AM-Q Gate）**：用近似模型 Q 函数门控离线 PPO 更新接受与否，保证单调改善；仅来源 50
- **迭代离线 RL（Iterative Offline RL + Dataset Expansion）**：每轮 RL→rollout→数据合并→IL 重训的飞轮循环；仅来源 50
- **一步一致性蒸馏（Consistency Model for Diffusion Policy Deployment）**：将多步 DDIM 策略压缩为一步 CM 用于高频部署；仅来源 50（在机器人领域的具体应用）

---

## 关联页面

- [[wiki/sources/rl-finetuning/2026-04-19 DPPO]] — 最接近方法：PPO 微调扩散策略
- [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] — Flow Matching 策略 RL 微调
- [[wiki/entities/systems/HIL-SERL]] — 核心比较系统
- [[wiki/analyses/RL微调生成模型的技术挑战与解决方案]] — 方法背景
