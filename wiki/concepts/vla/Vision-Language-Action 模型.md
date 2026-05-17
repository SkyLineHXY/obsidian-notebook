---
type: concept
tags: [VLA, 具身智能, 多模态, Foundation Model, 机器人策略, Pi系列, GR00T]
sources: [π₀.₆ 论文, LeRobot README, StarVLA README, RLinf README, Embodied-AI-Guide, wiki/sources/vla/2026-04-23 VLASH]
created: 2026-04-19
updated: 2026-04-23
---

# Vision-Language-Action 模型（VLA）

**Vision-Language-Action 模型（VLA）**是把**视觉观测 + 自然语言指令 + 机器人动作**三者统一到同一多模态 Transformer 中的大规模策略模型，是**"机器人基础模型"路线的主流形态**。用户通过自然语言 prompt 指定任务，VLA 直接输出动作序列。

---

## 核心思想

```
观测 o_t (图像 + 本体感知) ─┐
                          ├──▶  VLA（通常带 VLM 预训练骨干）──▶  a_t (或动作块)
语言指令 L ────────────────┘
```

- **骨干**：常基于 VLM（如 PaLI、Qwen-VL、Llama Vision），继承开放世界视觉-语言表征
- **动作头**：自回归 token、Diffusion、或 **[[wiki/concepts/generative-models/Flow Matching|Flow Matching]]**
- **训练数据**：大规模多机器人多任务演示数据（Open X-Embodiment 等）
- **部署**：通过 prompt 指定新任务，无需重新训练

---

## 代表性模型谱系

| 模型 | 出处 | 特点 |
|------|------|------|
| **RT-1 / RT-2**（Google） | 2022–2023 | VLA 早期代表 |
| **OpenVLA** | 2024 | 开源 7B VLA |
| **π₀ → π₀.₅ → [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP\|π₀.₆]]** | Physical Intelligence | 使用 Flow Matching 动作头；π₀.₆ 加入 advantage conditioning 支持 RL |
| **Pi0Fast / Pi0.5** | HuggingFace 复现 | 收录于 [[wiki/sources/frameworks/2026-04-19 LeRobot\|LeRobot]] |
| **GR00T N1.5** | NVIDIA | 人形机器人导向 |
| **SmolVLA** | 社区 | 轻量版 |
| **XVLA** | — | LeRobot 生态模型 |

---

## 在知识库中的出现

| 来源 | 覆盖 |
|------|------|
| [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] | **算法侧**：VLA 的 RL 微调范式；π₀.₆ + [[wiki/concepts/rl/RECAP\|RECAP]] |
| [[wiki/sources/frameworks/2026-04-19 LeRobot]] | **系统侧**：开源 VLA 统一部署框架；Pi0Fast、Pi0.5、GR00T N1.5、SmolVLA、XVLA |
| [[wiki/sources/frameworks/2026-04-19 StarVLA]] | **框架侧**：乐高式 VLA 开发框架；FAST/OFT/PI/GR00T 四变体；WM4A 世界模型融合 |
| [[wiki/sources/frameworks/2026-04-19 RLinf]] | **基础设施侧**：支持 π₀、GR00T、OpenVLA、[[wiki/entities/frameworks/StarVLA\|StarVLA]] 等多模型 RL 训练 |
| [[wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide]] | **知识体系侧**：VLA 在具身智能技术栈中的综述定位 |

---

## 关键研究问题

1. **如何用 RL 提升 VLA**？传统 PPO / REINFORCE 不易 scale，π₀.₆ 用 [[wiki/concepts/rl/RECAP|RECAP]]（advantage conditioning）规避 PG 训练；[[wiki/entities/frameworks/RLinf]] 提供统一 RL 训练基础设施。见 [[RL 微调表达性策略方法对比]]
2. **如何融合异构数据**？演示、autonomous rollout、expert intervention 需要统一训练管线
3. **真实世界奖励**：稀疏、模糊、含噪的任务成功信号
4. **跨形态泛化**：同一 VLA 在 Franka、SO-100、人形机器人上的一致性

---

## 与非 VLA 策略的对比

| 维度 | 传统模仿学习策略（如 [[wiki/concepts/generative-models/Diffusion Policy\|Diffusion Policy]]） | VLA |
|------|------|------|
| 任务指定 | 单任务或少量任务 | 自然语言 prompt（开放集） |
| 数据规模 | 百小时级 | 万小时级跨机器人 |
| 骨干 | 专用 CNN/Transformer | VLM 预训练骨干 |
| 泛化 | 同一平台、同分布 | 跨任务、跨硬件 |
| 推理成本 | 低 | 高（大模型开销） |

---

## 知识缺口
- **RT-2 / OpenVLA** 尚无独立出现的原始来源
