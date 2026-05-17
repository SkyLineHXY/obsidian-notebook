---
type: source
tags: [Embodied AI, Survey, Learning Resource, Community, Chinese]
sources: [raw/sources/github/TianxingChenEmbodied-AI-Guide Lumina具身智能社区 具身智能技术指南 Embodied-AI-Guide.md]
created: 2026-04-19
updated: 2026-04-19
---

[[raw/sources/github/TianxingChenEmbodied-AI-Guide Lumina具身智能社区 具身智能技术指南 Embodied-AI-Guide.md]]

# Embodied-AI-Guide — 具身智能技术指南

**GitHub**：https://github.com/TianxingChen/Embodied-AI-Guide  
**主创**：Tianxing Chen & Lumina 具身智能社区  
**定位**：国内最热门的具身智能中文百科全书式知识库（10k+ Stars，2025-12 突破）

---

## 定位与使用方法

该指南面向**具身智能新入门者**，采用"百科全书"形式：

- **实践先行**：以[[RoboTwin]]] 平台走通策略完整生命周期（数据采集 → 训练 → 评测）
- **宏观地图**：介绍各技术方向"能解决什么问题"，建立领域认知框架

**许可**：非商业使用（学术/个人研究允许，商业使用需联系作者）

---

## 动手教程：RoboTwin 2.0 路径（约 3.5 天）

1. **(～1 天)** 阅读 RoboTwin 2.0 论文，了解仿真数据合成方案，熟悉 Aloha 硬件
2. **(～0.5 天)** 安装 RoboTwin 2.0（基于 SAPIEN 仿真平台），采集 `beat_block_hammer` 任务 50 条数据
3. **(～1 天)** 复现 [[wiki/concepts/imitation-learning/ACT|ACT]] 策略（需 12GB 显存）
4. **(～1 天)** 评测：`demo_clean` 下 ACT 成功率约 **56%**（见 Leaderboard）

---

## 知识体系结构（7 章）

### (1) 入门
具身智能定义：基于物理实体感知与行动的智能系统，通过与环境交互产生适应性行为。

### (2) 动手学习（→ 上述 RoboTwin 路径）

### (3) 有用资料（快速建立认知）

- **社区**：石麻日记（强推）、Lumina具身智能、RLCN强化学习研究等公众号
- **实验室资源**：Robotics 实验室总结（知乎）、Awesome Humanoid Robot Learning（Yanjie Ze）
- **年度综述**：State of Robot Learning (Dec 2025)、许华哲 2025 回望、林天威 VLA 2025 综述
- **论文列表**：Awesome RL-VLA（Haoyuan Deng）、Awesome Efficient-VLA（Weifan Guan）
- **重要期刊/会议**：Science Robotics, TRO, IJRR, RSS, RAL, IROS, ICRA, CoRL, CVPR, NeurIPS, ICLR

### (4) 算法篇（技术栈从底层到顶层）

- 常用工程工具
- 视觉基础模型（2D/3D/4D Vision、Visual Prompting & Affordance）
- 机器人学习（RL/IL）
- LLM + 机器人
- [[wiki/concepts/vla/Vision-Language-Action 模型|VLA]]：经典工作 → 分层双系统 → 最新进展
- 计算机图形学

### (5) 软件基础设施篇

- **仿真器**：决定能构建什么样的世界
- **基准集**：决定如何比较方法优劣
- **数据集**：决定模型学到什么行为分布

### (6) 控制篇

- 经典/现代/先进控制理论
- 机器人学：运动学 & 动力学、里程计与 SLAM、工程生态（ROS 等）

### (7) 硬件篇

- 嵌入式、机械设计、机器人系统设计
- 传感器（深度相机）、触觉感知（视触觉传感器、电子皮肤）
- 数据采集硬件、公司生态

---

## 支持机构

无界智航、超维动力、香港大学 MMLab、地瓜机器人、松灵机器人

---

## 关联

-[[RoboTwin]]] — 指南入门教程的核心仿真平台
- [[wiki/concepts/vla/Vision-Language-Action 模型]] — 算法篇核心章节
- [[wiki/concepts/generative-models/Diffusion Policy]] — Robot Learning 章节的重要内容
- [[wiki/entities/hardware/Ubuntu 20.04]] — 硬件/系统配置基础（与来源 1-3 衔接）
