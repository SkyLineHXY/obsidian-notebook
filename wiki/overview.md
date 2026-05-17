# Overview

> **High-level synthesis** of the entire knowledge base. Updated periodically to reflect the current state of understanding across all ingested sources.

---

## Current Thesis

当前知识库围绕**将表达性机器人学习策略（Diffusion Policy、Flow Matching、VLA）部署到真实机械臂并通过强化学习持续提升**这一主线展开。相比首批摄取只覆盖"系统配置 + 单篇算法论文"，本次扩展（2026-04-19）使知识库涵盖了**算法 → 微调 → 大模型 → 开源框架**的完整纵切面。

---

## Key Themes

### 1. 系统环境配置（基础层）
三篇博客文章（来源 1–3）构成了一条清晰的**工程安装链**：

```
安装 Ubuntu 双系统 → 编译 PREEMPT_RT 实时内核 → 配置 NVIDIA 驱动
→ 安装 polymetis 控制框架 → 部署 Diffusion Policy
```

关键洞察：PREEMPT_RT 实时内核是连接操作系统与机械臂实时控制的**核心桥梁**，但会破坏标准 NVIDIA 驱动安装流程，需要手动编译内核模块。

### 2. 表达性机器人策略的两条主流路线

| 路线 | 代表工作 | 特点 |
|------|---------|------|
| **Diffusion Policy** | [[wiki/sources/imitation-learning/2026-04-19 Diffusion Policy (Chi 2024)]]（RSS 2023 / IJRR 2024）、[[wiki/sources/rl-finetuning/2026-04-19 DPPO]]（RSS 2025） | DDPM 去噪过程，成熟基线，15 任务相对 SOTA +46.9% |
| **[[wiki/concepts/generative-models/Flow Matching\|Flow Matching]]** | [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]]、[[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning\|FQL]]、[[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] | Rectified Flow，1 步推理，RL 微调更难但已被多种方案攻克 |

两者在机器人社区**并行发展**，共享"多模态动作分布、action chunking、长时序一致性"等设计决策。

### 3. RL 微调表达性策略的四种范式（**本次扩展的核心成果**）
详见 [[RL 微调表达性策略方法对比]]。共同技术难点：迭代生成策略的 BPTT 不稳定、边缘似然不可解析。

| 方法 | 解法主干 | 训练范式 |
|------|---------|---------|
| [[wiki/concepts/rl/DPPO\|DPPO]] | 把去噪链展开为内层 MDP + PPO | 在线 PG |
| [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow\|ReinFlow]] | 注入可学习噪声把 ODE 变 Markov 过程 | 在线 PG |
| [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning\|FQL]] | 蒸馏到一步策略，BC 与 Q 解耦 | 离线 RL（可 offline→online） |
| [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP\|RECAP / π₀.₆]] | Advantage conditioning 替代 PG | 迭代离线 RL，大 VLA |

**方法论分歧**：PG 到底适不适合表达性策略？三种回答（DPPO/ReinFlow 说"适合"、FQL 说"绕开更好"、RECAP 说"大模型上不 scale"）折射出领域活跃的开放问题。

### 4. VLA 与机器人基础模型
[[wiki/concepts/vla/Vision-Language-Action 模型]] 是当前机器人领域的另一条主线：
- **π₀.₆**（Physical Intelligence）：Flow Matching 动作头 + 迭代离线 RL，支持**叠衣、装纸箱、做咖啡**等复杂长时序真实任务
- **LeRobot**（HuggingFace）：Pi0Fast、Pi0.5、GR00T N1.5、SmolVLA、XVLA 的开源统一实现

### 5. 工程框架：从 polymetis 到 LeRobot
[[wiki/sources/imitation-learning/2026-04-18 复现Diffusion-policy模型]] 代表**重工程、Franka 专用**的经典路线（polymetis + libfranka + PREEMPT_RT）。[[wiki/sources/frameworks/2026-04-19 LeRobot|LeRobot]] 则代表**轻量、硬件无关、数据集中心**的新范式。**这回答了上一版 overview 的 Open Question #1**：polymetis 的更现代替代方案**就是 LeRobot**。

### 6. 知识库基础设施
[[wiki/sources/guides-tools/2026-04-19 MinerU Document Explorer|MinerU Document Explorer]] — 本知识库当前使用的文档解析基础设施，提供 PDF/Markdown 的目录导航、章节精读、公式提取等能力，遵循 Karpathy LLM Wiki 方法论。

---

## Open Questions

1. **PG vs 非 PG 的 scaling 曲线**：RECAP 声称 PPO 不 scale 到大 VLA，但 DPPO 在 FurnitureBench 长时序任务上用 PPO 成功——两者规模不同，真正的断点在哪？
2. **Sim2Real 的 Flow vs Diffusion 差异**：DPPO 报告了明显的 sim2real 成功，ReinFlow 尚未做真机实验，Flow 策略的真机表现仍待验证。
3. **LeRobot 生态对 Franka + polymetis 重工程栈的替代程度**：LeRobot 是否已有 Franka 原生支持？迁移成本如何？
4. **Advantage Conditioning 的泛化性**：RECAP 在 Flow Matching VLA 上成功，是否可迁移到 Diffusion 或非 VLA 场景？
5. **Rectified Flow 的理论最优性**：线性路径在所有任务上都是最好的吗，还是只是"够用"？

---

## Knowledge Gaps
详见 [[wiki/index|wiki/index.md]] 底部 "Knowledge Gaps" 小节。本轮更新后，以下概念已达 ≥2 来源阈值并建页：

- **[[wiki/concepts/imitation-learning/ACT|ACT]]**（Action Chunking Transformer）— LeRobot + Embodied-AI-Guide

仍待达标：RECAP 方法本身、advantage conditioning、VQ-BeT、TDMPC、HIL-SERL 等。

---

## Last Updated

2026-04-19 — Wiki 全面更新：
- **新增 Source 页**：MinerU Document Explorer、Flow Matching (Lipman 2023)
- **新增 Concept 页**：ACT（达 ≥2 来源阈值）
- **更新**：Flow Matching 概念页（添加原始论文引用）、LeRobot/Embodied-AI-Guide（添加 ACT 链接）
- **CLAUDE.md**: schema 升级到 v1.6（强制文档解析规范）
