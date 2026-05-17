---
type: concept
tags: [Benchmark, Robot Manipulation, Simulation, Imitation Learning, VLA, Evaluation]
sources: [LeRobot GitHub README (来源 9), VLASH 论文 (来源 18), RLinf 官方文档 (来源 19)]
created: 2026-04-24
updated: 2026-04-24
---

# LIBERO

**全称**：**LI**felong ro**B**ot l**E**a**R**ning (Or: Lifelong Robot Learning)  
**类型**：机器人操控仿真基准（Benchmark）  
**论文**：Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning"（NeurIPS 2023）  
**GitHub**：https://github.com/Lifelong-Robot-Learning/LIBERO

---

## 定位

LIBERO 是面向**机器人操控策略**评估的标准仿真基准，聚焦于：

- **长程任务（Long-Horizon Tasks）**：一次 episode 涉及多个子步骤，考验策略的时序组合能力
- **知识迁移（Knowledge Transfer）**：在不同任务套件间评估模型的泛化与持续学习能力
- **物理真实感**：基于 MuJoCo 仿真引擎，提供高质量视觉与接触物理

LIBERO 已成为 VLA 和模仿学习方法的**事实标准评测套件**之一，与 MetaWorld、ManiSkill 并列。

---

## 评测套件

LIBERO 包含 **4 个主评测子集**：

| 子集 | 考察重点 | 示例任务 |
|------|---------|---------|
| **LIBERO-Spatial** | 空间关系泛化（not/on/in/beside） | "把杯子放到碗的左边" |
| **LIBERO-Object** | 跨物体泛化（不同形状/材质） | "抓取黄色瓶子" |
| **LIBERO-Goal** | 目标条件泛化（相同物体，不同目标位置） | "把物体放进抽屉" |
| **LIBERO-Long** | 长程序列（4–6 步子任务链） | "开抽屉 → 取物体 → 关抽屉 → 放置" |

---

## 在知识库中的出现

### 来源 9：LeRobot
[[wiki/sources/frameworks/2026-04-19 LeRobot]] 将 LIBERO 列为标准评估环境之一：

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object
```

体现 LIBERO 已被 LeRobot 工具链直接集成，作为 VLA 微调效果的首选验证场景。

### 来源 18：VLASH
[[wiki/sources/vla/2026-04-23 VLASH]] 在 LIBERO 4 子基准上评估了 [[wiki/entities/models/π₀.₅]] + VLASH 异步推理方案：

| 方法 | 推理延迟 $\Delta$ | 平均成功率 | 加速 |
|------|----------------|-----------|-----|
| Sync（同步基线） | 0 | 96.8% | — |
| VLASH | 1 | 97.2% | **1.17×** |
| VLASH | 2 | 97.1% | **1.31×** |
| VLASH | 3 | 94.6% | **1.47×** |

表明高质量 VLA（π₀.₅）在 LIBERO 上已接近饱和（~97%），VLASH 能在保持精度的同时显著提速。

### 来源 19：RLinf 官方文档
[[wiki/sources/frameworks/2026-04-24 RLinf文档]] 将 LIBERO 列为 RLinf 内置支持的仿真评估环境之一（与 ManiSkill3、IsaacLab 并列），可通过 RLinf 框架直接运行 VLA 的在线 RL 微调实验。

---

## 与知识库中其他方法的联系

| 方法/模型 | LIBERO 用途 |
|---------|-----------|
| [[wiki/concepts/generative-models/Diffusion Policy]] | 早期在 LIBERO 子集上验证（Push-T 等较简单任务） |
| [[wiki/concepts/imitation-learning/ACT]] | LIBERO 是 ACT 系列方法的主要评测场景之一 |
| [[wiki/entities/models/π₀.₅]] | VLASH 实验以 LIBERO 为主基准，96.8% 同步基线 |
| [[wiki/entities/models/SmolVLA]] | LeRobot 中 SmolVLA 可用 LIBERO 场景做快速验证 |
| [[wiki/entities/frameworks/RLinf]] | LIBERO 作为 RL 微调的在线仿真环境 |

---

## 与同类基准的对比

| 基准           | 物理引擎           | 任务规模               | 特色                 |
| ------------ | -------------- | ------------------ | ------------------ |
| **LIBERO**   | MuJoCo         | 100 任务（LIBERO-100） | 长程 + 知识迁移，VLA 主流评测 |
| MetaWorld    | MuJoCo         | 50 任务              | 多任务操控，RL 经典基准      |
| ManiSkill3   | PhysX / Vulkan | 100+ 任务            | GPU 并行化渲染，RL 训练友好  |
| [[RoboTwin]] | SAPIEN         | 50 任务              | 双臂操控 + 合成数据生成      |

