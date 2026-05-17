---
type: source
tags: [扩散模型, 模仿学习, Visuomotor Policy, 机器人操作, RSS 2023, IJRR 2024]
sources: [raw/sources/papers/IL(Imitation Learning)/Chi 等 - 2024 - Diffusion Policy Visuomotor Policy Learning via Action Diffusion.pdf]
created: 2026-04-19
updated: 2026-04-19
---

[[Chi 等 - 2024 - Diffusion Policy Visuomotor Policy Learning via Action Diffusion.pdf]]

# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion

> **论文**：IJRR 2024（RSS 2023 会议扩展版） | arXiv:2303.04137v5
> **作者**：Cheng Chi、Zhenjia Xu、Siyuan Feng、Eric Cousineau、Yilun Du、Benjamin Burchfiel、Russ Tedrake、Shuran Song
> **单位**：Columbia University、Toyota Research Institute、MIT
> **项目主页**：https://diffusion-policy.cs.columbia.edu

---

## 研究问题

机器人从演示中学习（Learning from Demonstration, LfD）的核心挑战来自**机器人动作分布的特殊性**：
1. **多模态分布**：同一状态下可能存在多种合理动作（例如绕开障碍的左右两种路径），传统回归模型会坍缩到均值。
2. **序列相关性**：动作需保持时间一致，避免短视规划。
3. **高精度要求**：机器人动作的微小误差会被物理世界放大。

先前工作尝试用**混合高斯（MDN）**、**分类表示**、**隐式策略（EBM）**来应对，但各有训练不稳或表达力受限的问题。

---

## 核心贡献

### 算法：Diffusion Policy
将机器人策略建模为**条件去噪扩散过程（Conditional DDPM）**：从高斯噪声出发，通过 $K$ 步迭代去噪生成动作序列 $A_t = (a_t, a_{t+1}, \ldots, a_{t+T_a})$，以观测 $O_t$ 为条件。

**去噪更新式**：

$$
A_t^{k-1} = \alpha\left(A_t^k - \gamma\,\varepsilon_\theta(O_t, A_t^k, k)\right) + \mathcal{N}(0, \sigma^2 I)
$$

- $\varepsilon_\theta$：噪声预测网络（CNN 或 Transformer）
- 训练损失：$L = \mathrm{MSE}(\varepsilon^k,\; \varepsilon_\theta(O_t, A_t^0 + \varepsilon^k, k))$

### 三个关键技术决策

| 设计 | 作用 |
|------|------|
| **闭环动作序列（Receding Horizon Control）** | 预测 $T_p$ 步动作但仅执行 $T_a$ 步，再重新规划；兼顾长时序规划与即时反馈 |
| **视觉条件化（Visual Conditioning）** | 把视觉观测作为条件输入而非联合分布的一部分，只在每次推理时提取一次图像特征，显著降低计算量 |
| **时序扩散 Transformer** | 用带因果注意力的 Transformer 取代 CNN FiLM 主干，缓解过平滑、支持高频动作控制 |

### 两种网络架构
- **CNN-based**：1D U-Net + FiLM 条件注入，无需超参调优，作为首选基线
- **Transformer-based**：多头交叉注意力注入观测特征，在**高频动作变化**与**速度控制**任务上更强

---

## 实验结果

### 基准覆盖
- **15 个任务 / 4 个基准**：Robomimic（5 任务）、Push-T、Franka Kitchen（多阶段长时序）、IBC 的块推送
- **动作空间**：2–6 DoF
- **观测模式**：状态 + 图像
- **硬件**：模拟 + 真实（UR5、Franka、双臂系统）
- **扩展实验（IJRR 版新增）**：双臂打蛋器、展开桌垫、叠衬衫

### 核心数值
| 指标           | 结果              |
| ------------ | --------------- |
| 相对 SOTA 平均提升 | **+46.9%**      |
| 支持训练稳定性      | 基本无需任务特定调参      |
| 表达多模态分布      | 显著优于 MDN / 隐式策略 |

### 讨论新增章节（IJRR 扩展版）
- **与控制论的联系**：把去噪过程视为一种 Stochastic Langevin Dynamics
- **架构与预训练/微调范式的消融**：CNN vs Transformer 的适用场景、是否值得视觉骨干预训练

---

## 影响与后续工作

Diffusion Policy 成为表达性机器人策略的**事实标准基线**，催生了 RL 微调扩散策略的整个研究线：
- **[[wiki/sources/rl-finetuning/2026-04-19 DPPO|DPPO]]**（Ren 2024）：基于 PPO 的在线 RL 微调，把去噪步展开为内层 MDP
- **[[wiki/sources/rl-finetuning/2026-04-18 ReinFlow|ReinFlow]]**（Zhang 2025）：把同样思想扩展到 Flow Matching 策略
- **[[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning|Flow Q-Learning]]**（Park 2025）：用蒸馏规避迭代策略的 BPTT
- **[[wiki/sources/frameworks/2026-04-19 LeRobot|LeRobot]]**：HuggingFace 将 Diffusion Policy 纳入官方支持算法

---

## 关联知识
- 核心概念页：[[wiki/concepts/generative-models/Diffusion Policy]]
- 硬件相关：[[wiki/entities/hardware/Franka Research 3]]
- 工程复现：[[wiki/sources/imitation-learning/2026-04-18 复现Diffusion-policy模型]]
- RL 微调对比：[[RL 微调表达性策略方法对比]]
