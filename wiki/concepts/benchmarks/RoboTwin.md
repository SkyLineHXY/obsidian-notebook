---
type: entity
tags: [Simulation, Benchmark, Dual-Arm, Embodied AI, Platform]
sources: [RLinf README, StarVLA README, Embodied-AI-Guide]
created: 2026-04-19
updated: 2026-04-19
---

# RoboTwin

**类型**：双臂机器人仿真 & 基准平台  
**GitHub**：https://github.com/robotwin-Platform/RoboTwin  
**论文**：arXiv:2506.18088

---

RoboTwin 是基于 **SAPIEN** 仿真引擎开发的双臂机器人操作平台，提供：

- **50 个双臂任务**的自动化数据合成流程
- 主流操作策略（ACT、Diffusion Policy 等）的**训练测试集成**
- 完整的**策略评测系统**与公开 Leaderboard

RoboTwin 2.0 是当前社区推荐的**具身智能入门实践平台**（详见 [[wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide|具身智能技术指南]]），可在约 3.5 天内完成一个 VLA 策略的完整生命周期。

## 入门参考结果

- 任务：`beat_block_hammer`（50 条 demo_clean 数据）
- 策略：ACT（Action Chunking Transformer，需 ~12GB 显存）
- 成功率：**~56%**（demo_clean 评测条件）

## 在知识库中的角色

| 出现来源                                                             | 使用场景       |
| ---------------------------------------------------------------- | ---------- |
| [[wiki/sources/frameworks/2026-04-19 RLinf\|RLinf]]                         | RL 训练仿真器之一 |
| [[wiki/sources/frameworks/2026-04-19 StarVLA\|StarVLA]]                     | 基准测试环境     |
| [[wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide\|Embodied-AI-Guide]] | 入门实践教程核心平台 |

## 关联

- [[wiki/entities/frameworks/RLinf]] — RLinf 深度集成 RoboTwin 作为 RL 训练环境
- [[wiki/entities/frameworks/StarVLA]] — StarVLA 在 RoboTwin 上测评 VLA 性能
- [[wiki/concepts/generative-models/Diffusion Policy]] — RoboTwin 平台支持的策略类型之一
