---
type: entity
tags: [机械臂, Franka, 机器人硬件, 具身智能]
sources: [raw/sources/blogs/复现Diffusion-policy模型.md, raw/sources/papers/Zhang 等 - 2026 - ReinFlow Fine-tuning Flow Matching Policy with Online Reinforcement Learning.pdf]
created: 2026-04-18
updated: 2026-04-18
---

# Franka Research 3

**Franka Research 3**（FR3）是 Franka Emika 出品的 7-DOF 研究级机械臂，是当前知识库中机器人学习实验的**主要硬件平台**。

## 基本参数
- **制造商**：Franka Emika（现为 Agile Robots 旗下）
- **系统版本**（知识库中）：5.6.0
- **接口协议**：libfranka（C++ 底层驱动）

## libfranka 版本对应关系（重要！）
| 机械臂系统版本 | 所需 libfranka 版本 |
|--------------|-------------------|
| 5.6.0        | **0.13.3**        |
| 早期版本      | 0.9.0（polymetis 默认） |

> ⚠️ 版本不匹配会报错：`libfranka: Incompatible library version (server version: x, library version: x)`

## 在知识库中的出现
| 来源 | 用途 |
|------|------|
| [[wiki/sources/imitation-learning/2026-04-18 复现Diffusion-policy模型]] | 部署 Diffusion Policy 的真实机械臂平台 |
| [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] | Franka Kitchen 基准测试（虚拟环境中的 Franka 仿真） |

## 软件栈
```
libfranka 0.13.3（底层硬件驱动）
    ↓
polymetis（PyTorch 实时控制器，需 PREEMPT_RT 内核）
    ↓
高层策略（Diffusion Policy / Flow Matching Policy 等）
```

## 相关概念
- 控制框架：polymetis（需要 [[wiki/concepts/infrastructure/PREEMPT_RT实时内核]]）
- 策略算法：[[wiki/concepts/generative-models/Diffusion Policy]]
- 操作系统：[[wiki/entities/hardware/Ubuntu 20.04]]
