---
type: entity
tags: [RL Infrastructure, Embodied AI, Open Source, Training Framework]
sources: [RLinf GitHub README, StarVLA README]
created: 2026-04-19
updated: 2026-04-19
---

# RLinf

**类型**：开源 RL 训练基础设施  
**GitHub**：https://github.com/RLinf/RLinf  
**技术报告**：RLinf-USER（arXiv:2602.07837）

---

RLinf（Reinforcement Learning **Inf**rastructure）是面向**具身 AI 与智能体 AI** 的灵活可扩展开源 RL 框架。"inf" 双关 Infrastructure 与 Infinite——强调作为基础设施支柱，同时支持开放式无限学习。

## 核心特性

- **算法覆盖广**：PPO、GRPO、SAC、IQL、DAPO、RLPD、SAC-Flow、[[RECAP]]、DAgger、[[HG-DAgger]]、DSRL 等
- **模型兼容广**：[[wiki/concepts/vla/Vision-Language-Action 模型|π₀/π₀.₅]]、OpenVLA/OpenVLA-OFT、GR00T、[[StarVLA]]、Qwen-VL 系列等
- **环境支持广**：ManiSkill、LIBERO、[[RoboTwin]]、RoboVerse、CALVIN 等 12+ 仿真器；Franka、XSquare Turtle2 真实硬件
- **高吞吐量**：Hybrid Execution Mode 在具身 RL 下达 **2.434×** 吞吐提升
- **双后端**：FSDP + HuggingFace/SGLang/vLLM（快速原型）；Megatron + SGLang/vLLM（大规模）

## 与知识库其他内容的关系

- [[wiki/sources/frameworks/2026-04-19 RLinf]] — 来源摘要页（GitHub README）
- [[wiki/sources/frameworks/2026-04-24 RLinf文档]] — 官方文档摘要页（M2Flow、执行模式、RL 算法全览）
- [[wiki/sources/frameworks/2026-04-19 StarVLA]] — StarVLA 在 2026-04 集成 RLinf 做 RL 后训练
- [[wiki/concepts/rl/DPPO]] — RLinf 同时支持 DPPO 路线和 RECAP 路线的扩散策略 RL
- [[wiki/concepts/benchmarks/LIBERO]] — RLinf 内置支持的仿真评估环境
