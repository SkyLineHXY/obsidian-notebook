---
type: entity
tags: [VLA, HuggingFace, LeRobot, Compact Model, Open Source, Embodied AI]
sources: [LeRobot GitHub README (来源 9), VLASH 论文 (来源 18)]
created: 2026-04-24
updated: 2026-04-24
---

# SmolVLA

**类型**：紧凑型开源 Vision-Language-Action 模型  
**开发方**：HuggingFace  
**框架集成**：[[wiki/sources/frameworks/2026-04-19 LeRobot|LeRobot]]（原生支持）  
**模型家族**：Smol 系列（SmolLM → SmolVLM → SmolVLA）

---

## 定位

SmolVLA 是 HuggingFace **Smol 模型家族**在具身 AI 方向的延伸——用**轻量 VLM 骨干**（基于 SmolVLM）驱动机器人动作策略，目标是让 VLA 能够在**消费级硬件**上训练与推理，进一步降低机器人学习的准入门槛。

相比同类大型 VLA（[[wiki/entities/models/π₀.₅]]、GR00T 等），SmolVLA 的核心权衡是：

| 维度 | 大型 VLA（π₀.₅ 等） | SmolVLA |
|------|-------------------|---------|
| 参数规模 | 7B+ | 约 450M（SmolVLM 骨干） |
| 硬件需求 | 高端 GPU（A100/H100） | 消费级 GPU（RTX 3090 等） |
| 训练成本 | 高 | 低 |
| 通用泛化性 | 强 | 中 |
| 社区可及性 | 有限 | **LeRobot 开箱即用** |

---

## 在知识库中的出现

### 来源 9：LeRobot
[[wiki/sources/frameworks/2026-04-19 LeRobot]] 的"支持策略矩阵"中将 SmolVLA 列为 VLA 类原生支持模型之一（与 Pi0Fast、Pi0.5、GR00T N1.5、XVLA 并列）：

```bash
lerobot-train --policy=smolvla --dataset.repo_id=...
```

体现了 HuggingFace 将 SmolVLA 与 LeRobot 工具链深度整合、推动社区访问的战略。

### 来源 18：VLASH
[[wiki/sources/vla/2026-04-23 VLASH]] 将 SmolVLA 列于相关背景中，作为 VLA 异步推理框架的潜在适配对象之一（VLASH 方法本身在 [[wiki/entities/models/π₀.₅]] 上验证）。

---

## 与相关实体的关系

- **[[wiki/entities/models/π₀.₅]]**：同为 VLA，但 π₀.₅ 是 Physical Intelligence 的闭源大模型；SmolVLA 是开源轻量替代
- **[[wiki/entities/frameworks/RLinf]]**：RLinf 目前优先支持 OpenVLA、π₀、GR00T；SmolVLA 尚未明确列入，但 HuggingFace 生态兼容
- **[[wiki/entities/frameworks/StarVLA]]**：StarVLA 是乐高式 VLA 框架，SmolVLA 作为小型骨干可作为 StarVLA 某个变体的基础
- **[[wiki/concepts/vla/Vision-Language-Action 模型]]**：SmolVLA 是 VLA 范式的轻量化具体实现

---

## 知识缺口

以下信息尚需更多来源确认：
- SmolVLA 在具体基准（LIBERO、MetaWorld）上的精确性能数字
- 动作头设计（Flow Matching / Diffusion / ACT 分块？）
- 与 [[wiki/concepts/imitation-learning/ACT]] 相比的结构差异
