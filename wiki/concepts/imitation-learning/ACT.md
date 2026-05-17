---
type: concept
tags:
  - 模仿学习
  - Transformer
  - 动作分块
  - 机器人策略
  - Imitation Learning
  - CVAE
  - ALOHA
sources:
  - raw/sources/papers/IL(Imitation Learning)/Zhao 等 - 2023 - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.pdf
  - wiki/sources/frameworks/2026-04-19 LeRobot
  - wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide
  - wiki/sources/data-efficiency/2026-04-23 DemoSpeedup
  - wiki/sources/data-collection/2026-04-24 UMI (Chi 2024)
  - wiki/sources/data-collection/2026-04-24 FastUMI (Zhaxizhuoma 2025)
created: 2026-04-19
updated: 2026-04-24
---

# ACT (Action Chunking Transformer)

**Action Chunking Transformer (ACT)** 是一种基于 Transformer 的模仿学习策略，通过将动作序列分块（chunking）来实现高效的机器人行为学习。

## 核心思想

### 动作分块 (Action Chunking)

传统模仿学习方法通常预测单步动作，而 ACT 预测一个动作块（action chunk）—— 即未来 $k$ 个时间步的动作序列：

$$a_t, a_{t+1}, ..., a_{t+k-1} = \pi_\theta(o_t)$$

其中 $o_t$ 是当前观测（通常是视觉图像），$k$ 是块大小（chunk size）。

### Transformer 架构

ACT 使用标准的 Transformer 编码器-解码器架构：

- **编码器**: 处理视觉观测（ResNet 或 ViT 特征提取器）
- **解码器**: 生成动作序列（通过因果自回归）

## 关键技术细节

### 时序集成 (Temporal Ensembling)

为避免相邻动作块之间的抖动，ACT 使用时序集成：

$$
\bar{a}_t = \frac{\sum_{i=0}^{k-1} w_i \cdot a_{t}^{(t-i)}}{\sum_{i=0}^{k-1} w_i}
$$

其中 $a_{t}^{(t-i)}$ 是在时间步 $t-i$ 预测的、包含时间步 $t$ 的动作块中的对应动作。

### CVAE 风格训练

ACT 采用条件变分自编码器（CVAE）风格的训练：

1. **编码器**: 将历史动作序列编码为风格变量 $z$
2. **解码器**: 以 $z$ 和当前观测为条件，生成动作序列
3. **KL 散度**: 约束 $z$ 接近标准正态分布

目标函数：

$$\mathcal{L} = \underbrace{\mathbb{E}_{q(z|a_{1:T}, o)}[\log p(a_{1:T}|o, z)]}_{\text{重构损失}} + \beta \cdot \underbrace{D_{\text{KL}}(q(z|a_{1:T}, o) \| p(z))}_{\text{KL 正则化}}$$

## 性能特点

| 指标 | 数值 |
|------|------|
| 显存需求 | ~12 GB |
| RoboTwin demo_clean 成功率 | ~56% |

## 与 Diffusion Policy 的对比

| 特性 | ACT | Diffusion Policy |
|------|-----|------------------|
| **生成方式** | 自回归 Transformer | 扩散迭代去噪 |
| **多模态处理** | CVAE 隐变量 | 扩散过程的随机性 |
| **训练稳定性** | 较稳定 | 需要更多调参 |
| **高频控制** | 通过动作分块实现 | 同样支持动作分块 |
| **表达性** | 受限于 Transformer 容量 | 理论上可建模任意分布 |

## 实现要点

### 关键超参数

- **chunk_size**: 动作块大小（通常 10-100）
- **hidden_dim**: Transformer 隐藏维度
- **num_layers**: Transformer 层数
- **num_heads**: 注意力头数
- **beta**: KL 散度权重

### 数据要求

- 需要高质量的示范轨迹
- 动作频率应与任务时间尺度匹配
- 观测-动作对齐至关重要

## 应用场景

ACT 特别适合：
- **高频精细操作**（如装配、折叠）
- **需要平滑轨迹的任务**（动作分块天然提供平滑性）
- **数据受限场景**（相比扩散模型，ACT 参数量更小）

## Related Concepts

- [[wiki/concepts/generative-models/Diffusion Policy]] — 另一种高表达性模仿学习方法
-[[RoboTwin]]] — ACT 常用的仿真基准平台
- [[wiki/sources/frameworks/2026-04-19 LeRobot]] — 包含 ACT 实现的开源框架
- [[wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide]] — ACT 实验教程与基准评测

## References

- **原始论文**：[[wiki/sources/imitation-learning/2026-04-21 ACT - ALOHA (Zhao 2023)]]（Zhao et al., ICRA 2023）
- LeRobot 实现：[[wiki/sources/frameworks/2026-04-19 LeRobot]]
- Embodied-AI-Guide 教程：[[wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide]]
- DemoSpeedup 应用：[[wiki/sources/data-efficiency/2026-04-23 DemoSpeedup]]（以 ACT 为代理策略 + 最终策略，实现 2.1× 示教加速）

---

*本页遵循 [[CLAUDE.md]] 的 Wiki Schema 规范创建，满足 ≥2 来源阈值。*
