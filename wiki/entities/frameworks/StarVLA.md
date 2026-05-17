---
type: entity
tags: [VLA, Framework, Open Source, Modular, Robot Learning]
sources: [StarVLA GitHub README, RLinf GitHub README]
created: 2026-04-19
updated: 2026-04-19
---

# StarVLA

**类型**：开源 VLA 开发框架  
**GitHub**：https://github.com/starVLA/starVLA  
**Technical Report**：arXiv:2604.05014  
**创始人**：Jinhui Ye & Weiyu Guo

---

StarVLA 是一个以**乐高式（Lego-like）模块化设计为核心的 VLA 研究平台，"StarVLA" 同时是 "start VLA" 的双关。

## 四种动作头变体

| 变体 | 动作机制 | 对标 |
|------|---------|-----|
| FAST | 自回归离散 Token | π₀-fast |
| OFT | 并行 MLP | OpenVLA-OFT |
| PI | [[wiki/concepts/generative-models/Flow Matching]] | [[wiki/concepts/vla/Vision-Language-Action 模型\|π₀]] |
| GR00T | 双系统（VLM + Flow Matching） | GR00T N1 |

## 关键集成

- **RL 后训练**：通过 [[RLinf]] 支持（2026-04）
- **WM4A**：Cosmos-Predict2 / Wan2.2 视频 DiT 作为动作预测 backbone
- **VLM backbone**：Qwen3.5（0.8B~9B）、Florence-2、ABot-M0

## 与知识库其他内容的关系

- [[wiki/sources/frameworks/2026-04-19 StarVLA]] — 来源摘要页
- [[wiki/concepts/vla/Vision-Language-Action 模型]] — StarVLA 实现了 π₀、GR00T 等多个 VLA 架构
- [[wiki/concepts/generative-models/Flow Matching]] — StarVLA-PI / GR00T 变体的核心
- [[wiki/entities/frameworks/RLinf]] — RL 后训练后端
- [[wiki/entities/hardware/Franka Research 3]] — StarVLA 真机实验平台
