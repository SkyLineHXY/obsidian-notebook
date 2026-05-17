---
type: entity
tags: [Real-World RL, Dexterous Manipulation, Human-in-the-Loop, System, UC Berkeley]
sources: [LeRobot README (HuggingFace), Luo 等 2025 HIL-SERL 论文]
created: 2026-04-24
updated: 2026-04-24
---

# HIL-SERL

**HIL-SERL**（Human-in-the-Loop Sample-Efficient Robotic Reinforcement Learning）是 UC Berkeley（Jianlan Luo、Sergey Levine 等）开发的**真实世界机器人 RL 系统**，专注于通过人类在线纠错结合 off-policy RL 高效习得精密灵巧操控技能。

---

## 定位与背景

HIL-SERL 是 **SERL**（Luo 等 2024 ICRA）的进化版本：
- **SERL**：使用离线演示初始化的真实世界 off-policy RL
- **HIL-SERL**：在 SERL 基础上加入**在线人类纠错（corrections）**，解锁了 SERL 无法处理的高难度任务

项目网站：https://hil-serl.github.io/

---

## 核心组件

| 组件             | 实现                         |
| -------------- | -------------------------- |
| **底层 RL 算法**   | RLPD（SAC 变体，支持 prior data） |
| **视觉骨干**       | ResNet-10（ImageNet 预训练）    |
| **奖励函数**       | 二分类器（稀疏奖励，精度 >95%）         |
| **演示缓冲**       | 20–30 条专家轨迹（离线）            |
| **纠错机制**       | SpaceMouse 实时介入，数据写入双缓冲区   |
| **Gripper 控制** | 独立 DQN Critic（离散动作）        |

---

## 性能表现

- 在 **13 个** 真实机器人操控任务上训练 1–2.5 小时达到 **100% 成功率**（BC 基线平均 49.7%）
- 平均执行周期比 BC 快 **1.8×**
- 解决的任务包括：正时皮带装配（**6h 训练，2% → 100%**）、Jenga 抽块（**8% → 100%**）、双臂仪表板装配、RAM 插入等

---

## 与 HG-DAgger 的核心区别

HIL-SERL 的人类介入机制形式上与 [[wiki/concepts/imitation-learning/HG-DAgger]] 相同（仅在策略即将失败时接管），但本质不同：

| 维度 | [[wiki/concepts/imitation-learning/HG-DAgger]] | HIL-SERL |
|------|-----------------------------|----------|
| 纠错数据利用方式 | 监督学习（BC）| 强化学习（RL） |
| 能否超越人类演示 | 否 | **是**（RL 探索+动态规划）|
| 复杂任务成功率 | 受限于演示质量 | 实验中显著优于 DAgger |

---

## 技术特色：Q-value 漏斗

HIL-SERL 通过 RL 动态规划在状态空间中形成**漏斗结构**：
- 策略将从初始状态到目标的路径"鲁棒化"
- 关键状态表现为高 Q 值 + 高 Q 值方差（对扰动敏感）
- 这使策略能自发习得**外部扰动后的恢复行为**（重抓取、重对准等）

---

## 出现来源

- [[wiki/sources/frameworks/2026-04-19 LeRobot]] — 在 LeRobot 支持的方法列表中提及 HIL-SERL
- [[wiki/sources/rl-finetuning/2026-04-24 HIL-SERL]] — 原始论文完整介绍
