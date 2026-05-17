---
type: source
tags: [生成模型, Flow Matching, 扩散模型, 连续归一化流, ICLR]
sources: [raw/sources/papers/Generative Model/Lipman 等 - 2023 - Flow Matching for Generative Modeling.pdf]
created: 2026-04-19
updated: 2026-04-19
---

# Flow Matching for Generative Modeling (Lipman et al., 2023)

[[Lipman 等 - 2023 - Flow Matching for Generative Modeling.pdf]]

> **Published**: ICLR 2023  
> **Authors**: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le  
> **arXiv**: [2210.02747](https://arxiv.org/abs/2210.02747)

## 核心贡献

本文提出 **Flow Matching (FM)**，一种基于 Continuous Normalizing Flows (CNFs) 的生成建模新范式，实现了前所未有的训练规模。

### 主要创新点

1. **Simulation-Free Training**  
   FM 通过回归固定条件概率路径的向量场来训练 CNFs，无需模拟（simulation-free）。

2. **通用概率路径框架**  
   FM 兼容高斯概率路径族，包含现有扩散路径作为特例。

3. **最优传输 (OT) 路径**  
   使用最优传输位移插值定义条件概率路径，比扩散路径更高效。

## 方法概述

### 核心公式

Flow Matching 目标函数：

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x} \left\| v_\theta(t, x) - u_t(x) \right\|^2$$

其中 $v_\theta$ 是学习到的向量场，$u_t$ 是目标向量场。

### 条件 Flow Matching

通过条件概率路径 $p_t(x|x_1)$ 定义目标向量场：

$$u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1) p(x_1)}{p_t(x)} dx_1$$

条件向量场 $u_t(x|x_1)$ 通过高斯概率路径解析计算。

## 实验结果

- **训练效率**: 比扩散模型更快收敛
- **采样速度**: 使用 OT 路径实现更高效的采样
- **泛化性能**: 更好的泛化能力

## 与相关工作的关系

- **扩散模型**: FM 包含扩散路径作为特例，但更加通用
- **连续归一化流**: FM 提供了一种可扩展的 CNF 训练方法
- **最优传输**: FM 可直接利用 OT 路径的优势

## Related Concepts

- [[wiki/concepts/generative-models/Flow Matching]] — 本知识库的 Flow Matching 概念页，整合了本论文与 ReinFlow、Flow Q-Learning 等后续工作
- [[wiki/concepts/generative-models/Diffusion Policy]] — 扩散模型在机器人策略中的应用
- [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] — 使用 Flow Matching 进行强化学习微调
- [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]] — 基于 Flow Matching 的 Q 学习

---

*本页遵循 [[CLAUDE.md]] 的 Wiki Schema 规范创建。*
