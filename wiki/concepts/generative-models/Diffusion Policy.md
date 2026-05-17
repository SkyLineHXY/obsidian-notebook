---
type: concept
tags: [模仿学习, 扩散模型, 机器人控制, 策略学习, Visuomotor Policy]
sources: [raw/sources/papers/IL(Imitation Learning)/Chi 等 - 2024 - Diffusion Policy Visuomotor Policy Learning via Action Diffusion.pdf, raw/sources/blogs/复现Diffusion-policy模型.md, raw/sources/papers/VLA+RL/Zhang 等 - 2026 - ReinFlow Fine-tuning Flow Matching Policy with Online Reinforcement Learning_1.pdf, raw/sources/papers/VLA+RL/Ren 等 - 2024 - Diffusion Policy Policy Optimization.pdf, raw/sources/github/huggingfacelerobot 🤗 LeRobot Making AI for Robotics more accessible with end-to-end learning.md, wiki/sources/data-efficiency/2026-04-23 DemoSpeedup, wiki/sources/data-collection/2026-04-24 UMI (Chi 2024), wiki/sources/data-collection/2026-04-24 FastUMI (Zhaxizhuoma 2025)]
created: 2026-04-18
updated: 2026-04-24
---

# Diffusion Policy

**Diffusion Policy**（Chi et al., RSS 2023 / IJRR 2024）是一种将 **[[wiki/concepts/generative-models/DDPM|DDPM]]**（Ho et al., NeurIPS 2020）应用于机器人动作生成的**模仿学习**框架，将策略学习形式化为条件去扩散过程。它已成为**表达性机器人策略的事实标准基线**。

---

## 核心思想

将机器人动作序列 $A_t = (a_t, \ldots, a_{t+T_a})$ 建模为**条件去噪扩散过程的输出**，以当前观测 $O_t$（状态或图像）为条件：

$$
A_t^{k-1} = \alpha\bigl(A_t^k - \gamma\,\varepsilon_\theta(O_t, A_t^k, k)\bigr) + \mathcal{N}(0, \sigma^2 I)
$$

从高斯噪声 $A_t^K$ 出发，经过 $K$ 步迭代去噪生成动作序列。

**训练损失**（原论文 Eq. 5）：

$$
L = \mathrm{MSE}\bigl(\varepsilon^k,\; \varepsilon_\theta(O_t, A_t^0 + \varepsilon^k, k)\bigr)
$$

---

## 三个关键技术决策（原论文）

| 设计 | 作用 |
|------|------|
| **闭环动作序列** | 预测 $T_p$ 步、执行 $T_a$ 步，再重新规划（Receding Horizon Control） |
| **视觉条件化** | 观测作为条件而非联合分布一部分，每次只提取一次图像特征 |
| **时序扩散 Transformer** | 带因果注意力的 Transformer 支持高频动作控制 |

两种网络架构：
- **CNN-based (1D U-Net + FiLM)**：无需超参调优，首选基线
- **Transformer-based**：高频动作 / 速度控制任务更强

---

## 优势与局限

**优势**：
- **多模态动作分布**（解决传统 BC 的均值回归问题）
- **视觉输入原生支持**（Visuomotor Policy）
- **长动作序列生成**（Action Chunking）
- **训练稳定**，几乎不需任务特定调参
- 在 15 任务 / 4 基准上相对 SOTA **平均提升 46.9%**

**局限**：
- **推理慢**（需多步去噪）
- 依赖**专家数据质量**，无内置探索机制 → 催生 RL 微调方向

---

## 与 Flow Matching 的关系

Diffusion Policy 使用 **DDPM/DDIM** 扩散过程；**[[wiki/concepts/generative-models/Flow Matching|Flow Matching]]** 策略（Rectified Flow、Shortcut Models）使用更直的 ODE 路径，推理更快但 RL 微调更难。两者目前在机器人社区并行发展，是**表达性策略的两条主流路线**。

---

## RL 微调进展

Diffusion Policy 的性能受限于演示数据质量，促使大量 RL 微调工作。详细对比见 [[RL 微调表达性策略方法对比]]。

| 方法 | 目标 | 关键机制 |
|------|------|---------|
| [[wiki/concepts/rl/DPPO\|DPPO]]（RSS 2025） | 在线 RL | 把去噪展开为内层 MDP + PPO |
| [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow\|ReinFlow]]（NeurIPS 2025） | 在线 RL | 同类思想迁移到 Flow Matching |
| [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning\|FQL]]（ICML 2025） | 离线 RL | 蒸馏避免 BPTT |
| [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP\|RECAP]]（2025-11） | Offline RL（VLA 规模） | Advantage conditioning |

---

## 在知识库中的出现

| 来源 | 角色 |
|------|------|
| [[wiki/sources/generative/2026-04-21 DDPM (Ho 2020)]] | **生成骨干**：DDPM 原文，Diffusion Policy 的数学基础 |
| [[wiki/sources/imitation-learning/2026-04-19 Diffusion Policy (Chi 2024)]] | **原始论文**：方法、架构、15 任务评估、sim2real 扩展 |
| [[wiki/sources/imitation-learning/2026-04-18 复现Diffusion-policy模型]] | **工程实践**：在 [[wiki/entities/hardware/Franka Research 3\|Franka Research 3]] 上部署的完整栈 |
| [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] | **对比基线**：与 Flow Matching 策略对比 |
| [[wiki/sources/rl-finetuning/2026-04-19 DPPO]] | **RL 微调主体**：PPO 微调扩散策略 |
| [[wiki/sources/frameworks/2026-04-19 LeRobot]] | **社区实现**：作为 HuggingFace LeRobot 官方 IL 基线之一 |

---

## 相关资源
- **原始项目**：https://diffusion-policy.cs.columbia.edu
- **作者**：Cheng Chi、Zhenjia Xu、Shuran Song 等（Columbia、TRI、MIT）
- **训练环境**：robodiff（`conda_environment_real.yaml`）
- **硬件平台示例**：[[wiki/entities/hardware/Franka Research 3]]、UR5
- **数据采集**：Spacemouse 遥操作 + Intel RealSense D455
- **开源实现**：[Yingdong-Hu/diffusion_policy_vila](https://github.com/Yingdong-Hu/diffusion_policy_vila)、[[wiki/sources/frameworks/2026-04-19 LeRobot|LeRobot]] 内置版本
