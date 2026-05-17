---
type: concept
tags: [扩散策略, 在线强化学习, PPO, Policy Gradient, 双层 MDP]
sources: [raw/sources/papers/VLA+RL/Ren 等 - 2024 - Diffusion Policy Policy Optimization.pdf, raw/sources/papers/VLA+RL/Zhang 等 - 2026 - ReinFlow Fine-tuning Flow Matching Policy with Online Reinforcement Learning_1.pdf]
created: 2026-04-19
updated: 2026-04-23
---

# DPPO（Diffusion Policy Policy Optimization）

**DPPO**（Ren 等 2024，arXiv:2409.00588）是**用策略梯度 / PPO 在线微调预训练扩散策略**的算法框架。它是 [[wiki/concepts/generative-models/Diffusion Policy|Diffusion Policy]] 的 RL 微调**事实标准基线**，也是 [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow|ReinFlow]] 等后续方法的主要对比对象。

> **完整数学推导**（零基础向）见：[[wiki/analyses/DPPO 完整数学推导]]

---

## 核心思想：双层 MDP

把扩散策略的**去噪过程**本身视作一个 MDP（**Diffusion MDP**），嵌入到环境交互的外层 MDP：

```
Environment MDP (s_t, a_t, r_t)
         ↓ 每步动作由去噪过程产生
Diffusion MDP (a^K → a^{K-1} → … → a^0)
```

这样：
- 每个去噪步都有**可处理的高斯似然** $\log \pi_\theta(a^{k-1} \mid a^k, o)$
- 可以把 PPO 的 clip / advantage 估计直接套在**两层 MDP 的联合轨迹**上

**PPO 更新目标**（简化版）：

$$
\max_\theta\; \mathbb{E}\biggl[\min\Bigl(\rho_\theta \hat A,\; \mathrm{clip}(\rho_\theta, 1-\epsilon, 1+\epsilon)\hat A\Bigr)\biggr], \quad \rho_\theta = \prod_{k} \frac{\pi_\theta(a^{k-1} \mid a^k, o)}{\pi_{\theta_{\mathrm{old}}}(a^{k-1} \mid a^k, o)}
$$

---

## 关键设计决策

| 决策                        | 作用                                      |
| ------------------------- | --------------------------------------- |
| **PPO 作外层优化器**            | 相比 REINFORCE / Q-learning，在长时序、高维动作下更稳定 |
| **只微调最后几步去噪** 或换 **DDIM** | 减少有效时域，样本效率↑                            |
| **修改噪声 schedule**         | 微调阶段保留足够随机性，既是探索源也防 value collapse      |

---

## 理论贡献：结构化探索

DPPO 通过消融证明：相比高斯 / GMM 策略，**扩散策略在 RL 微调阶段沿训练数据流形（manifold）探索**，于是：
- 探索更"有方向"，不会乱撞障碍
- 微调后的策略**对观测扰动更鲁棒**
- 展现出显著的 **sim-to-real zero-shot** 能力（FurnitureBench → 真机）

这驳斥了先前 Psenka 等工作**"策略梯度对扩散策略无效"** 的推测。

---

## 在知识库中的出现

| 来源                                   | 角色                                                                                             |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| [[wiki/sources/rl-finetuning/2026-04-19 DPPO]]     | **主体**：DPPO 的完整算法与实验                                                                           |
| [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] | **对比基线**：DPPO 针对 Diffusion，ReinFlow 把同类思想迁移到 Flow Matching；ReinFlow 报告相对 DPPO 节省 62.82% 墙上时钟时间 |

---

## 与其他 RL-for-Generative-Policy 方法的关系

详见对比页：[[RL 微调表达性策略方法对比]]

| 维度          | **DPPO**   | ReinFlow         | FQL       | RECAP                         |
| ----------- | ---------- | ---------------- | --------- | ----------------------------- |
| 策略类         | Diffusion  | Flow             | Flow      | Flow VLA                      |
| 训练范式        | 在线 RL（PG）  | 在线 RL（PG）        | 离线 RL（蒸馏） | 离线 RL（advantage conditioning） |
| 是否 BPTT 去噪链 | 否（用每步高斯似然） | 否（噪声注入化为 Markov） | 否（蒸馏避免）   | 否（conditioning 避免）            |
| 推理开销        | 多步 / DDIM  | 支持 1 步           | 单步        | 多步 Flow                       |

---

## 项目信息
- **代码主页**：https://diffusion-ppo.github.io
- **作者**：Princeton、MIT、TRI、CMU
