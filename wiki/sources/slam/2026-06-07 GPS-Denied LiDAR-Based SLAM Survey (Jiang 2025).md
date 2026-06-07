---
type: source
tags: [SLAM, LiDAR, Survey, SensorFusion, StateEstimation, Robotics, KalmanFilter]
sources: [raw/sources/papers/其他/Jiang 等 - 2025 - GPS-Denied LiDAR-Based SLAM—A Survey/Jiang 等 - 2025 - GPS-Denied LiDAR-Based SLAM—A Survey.md]
created: 2026-06-07
updated: 2026-06-07
---

# GPS-Denied LiDAR-Based SLAM — A Survey (Jiang et al. 2025)

> 原文回链：[[raw/sources/papers/其他/Jiang 等 - 2025 - GPS-Denied LiDAR-Based SLAM—A Survey/Jiang 等 - 2025 - GPS-Denied LiDAR-Based SLAM—A Survey.md]]
> 出处：*IET Cyber-Systems and Robotics*（开放获取），2025；Received Jan 2025 / Accepted Aug 2025。
> 作者：Haolong Jiang, Yikun Cheng, Weichen Dai, Wenbin Wan, Qinyao Liu, Fanxin Wang（西交利物浦 / UIUC / 杭电 / UNM）。

## 一句话总结

一篇面向 **GPS 拒止环境**（室内、地下、隧道、矿井、森林）的 **LiDAR 为中心的多传感器融合 SLAM** 综述，重点放在 **滤波式（filter-based）传感器融合**，系统梳理了硬件/数据集、定位（前向传播 + 后向传播）、建图（关键帧/数据结构/回环/渲染）三大模块，并以 FAST-LIO 为典型范式给出完整状态估计数学链条。

## 背景与定位

- 传统综述多聚焦 **视觉 SLAM** 或 **GPS 融合 SLAM**；本文专攻 GPS 不可用 + 高动态 / 稀疏特征 / 极端光照的场景。
- 视觉 SLAM（[[wiki/entities/hardware/UMI]] 用到的 ORB-SLAM3 即属此类）依赖光照与纹理，在隧道、矿井、夜间失效；LiDAR SLAM 凭借主动测距对光照鲁棒，是 GPS 拒止环境的首选。
- 选择滤波式（EKF/UKF/PF）而非纯优化式，理由是其 **实时性与工业落地价值**（VIO/无人车导航的递归状态估计）。

## 整体框架

![[raw/sources/papers/其他/Jiang 等 - 2025 - GPS-Denied LiDAR-Based SLAM—A Survey/images/a2efaca7cedffb67954d4bf7e6ce85292b99304cd8b997e91e87f5775b1f4702.jpg]]
*Figure 1：SLAM 总体框架。上半为定位（Localisation）：Sensor→{IMU 前向传播 / LiDAR 后向传播}→残差计算→优化（传统 EKF/UKF 或创新 L1-based/模型约束）→状态更新；下半为建图（Mapping）：关键帧选择→点插入/地图栈→回环检测→渲染→地图。*

## 1. 硬件与数据集

四类典型传感器组合（Table 1）：

| 组合 | 优势 | 劣势 | 代表算法 |
|---|---|---|---|
| 2D LiDAR + IMU | 结构简单、低成本、平面环境 | 无 3D 信息、环境适应弱 | Hector SLAM, GMapping |
| Camera + IMU | 低成本、纹理丰富场景有效 | 强依赖光照/纹理 | ORB-SLAM, VINS-Mono |
| 3D LiDAR + IMU | 高精度 3D 点云、复杂环境 | 高成本、点云计算密集 | LOAM, FAST-LIO |
| 3D LiDAR + Camera + IMU | 彩色点云 + 高精度、适应性最强 | 集成复杂、功耗高 | V-LOAM, LVI-SAM, FAST-LIVO |

- **数据集**（Table 2）：2D 平面导航（MIT Stata、ROS TurtleBot）；视觉惯性（EuRoC MAV、TUM VI、KITTI）；3D LiDAR（Newer College、NCLT、MulRan 隧道/城市）；多传感器（Complex Urban、Apollo-SouthBay、NTU-VIRAL、R3LIVE）。
- **更新方案两分**：**优化式**（重投影误差/BA/branch-and-bound，精度高但算力大）vs **滤波式**（EKF/UKF/PF 递归估计，实时性强，本文主线）。

## 2. 定位（Localisation）：前向 + 后向传播

以 **FAST-LIO** 为典型例。

**前向传播（Forward Propagation）**——用 IMU 在 LiDAR 帧间做高频递推：
- IMU 模型：$\boldsymbol{a}_m = \boldsymbol{a}_t + \boldsymbol{b}_a + \boldsymbol{n}_a$，$\boldsymbol{\omega}_m = \boldsymbol{\omega}_t + \boldsymbol{b}_\omega + \boldsymbol{n}_\omega$（含 bias + 白噪声 → 随机游走）。
- 连续运动学：$\dot{\boldsymbol{p}}=\boldsymbol{v}$，$\dot{\boldsymbol{v}}=\boldsymbol{R}(\boldsymbol{a}_m-\boldsymbol{b}_a-\boldsymbol{n}_a)+\boldsymbol{g}$，$\dot{\boldsymbol{R}}=\boldsymbol{R}\lfloor\boldsymbol{\omega}_m-\boldsymbol{b}_\omega-\boldsymbol{n}_\omega\rfloor^\wedge$。
- 协方差传播：$\boldsymbol{P}_k = \boldsymbol{F}_{k-1}\boldsymbol{P}_{k-1}\boldsymbol{F}_{k-1}^\mathrm{T} + \boldsymbol{G}_{k-1}\boldsymbol{Q}_{k-1}\boldsymbol{G}_{k-1}^\mathrm{T}$。

**后向传播（Backward Propagation）**——用 LiDAR 观测修正 IMU 累积漂移：
- 残差计算：通过 ICP / scan-to-model 匹配建立点对，残差 $\boldsymbol{r}_i = \boldsymbol{p}_i^\mathrm{o} - \boldsymbol{p}_i^\mathrm{p}$（观测点 − 预测点），用 KD-tree 做最近邻搜索。
- 优化分两类：**传统优化**（EKF/UKF 一阶泰勒线性化残差，增量更新 $\boldsymbol{x}_k = \boldsymbol{x}_k + \Delta\boldsymbol{x}$）；**创新优化**（紧耦合联合估计，如 UAV 弹性状态估计器，含 $L_1$ 自适应 + 约束投影，对抗 GPS 欺骗攻击）。

> 关键洞察：相机 SLAM 通常 **缺乏严格的后向传播**，其位姿优化依赖前端特征匹配 + 后端图优化，难以像 LiDAR SLAM 那样用点云残差显式修正——这是 LiDAR SLAM 在 GPS 拒止下的独特优势。完整推导见 [[wiki/concepts/slam/SLAM (Simultaneous Localization and Mapping)]]。

## 3. 建图（Mapping）

- **关键帧选择**：阈值式（ORB-SLAM2）、运动估计式（VINS-Mono/MSCKF）、**信息增益式**（LIO-SAM，用 Fisher 信息矩阵 $\boldsymbol{I}(\boldsymbol{x})=\sum_i \boldsymbol{H}_i^\mathrm{T}\boldsymbol{R}_i^{-1}\boldsymbol{H}_i$ 量化每帧贡献）、深度/强化/自监督/GNN 式。
- **数据结构**：KD-tree → **IKD-tree**（增量更新，FAST-LIO2 核心）→ Octree / i-Octree → Voxel grid / voxel hashing。各有时空权衡（Table 8）。
- **回环检测**：图优化（pose graph）、scan matching（Siamese 网络）、**BoW / BoW3D**（6-DoF 实时回环）、深度学习（L3D 描述子）、强化学习。
- **渲染**：LiDAR-only 点云（ToF 测距 $d = c\Delta t / 2$ + 投影）；LiDAR + Camera 彩色点云；新趋势 **3DGS / NeRF**（精度极高、速度低）。

## 4. 实验对比（亮点）

- 作者在 **Livox Mid-360 + Jetson Orin Nano** 手持系统上实测办公楼三层楼梯/大厅 3D 重建。
- 提出 **新评估指标**：扫描墙面厚度、楼面/楼梯墙倾角、物体半径误差（用 0.75 m 真值吊灯标定）。
- **Faster-LIO** 综合最优（表面厚度 4.58 cm、倾角最小、半径误差 +0.03 m）；APE 上 FAST-LIO2 / Inv-DLIO 在多数序列领先（Table 12/13）。

## 5. 开放问题与未来方向

1. **GPS 拒止解法**：AI 信号去噪（CNN 实时剔除不可靠 GPS 段）；**弹性状态估计**（对抗 GPS 欺骗——攻击者定位跟踪 ALT + 逃逸控制器 ESC，在 escape time 内驶离欺骗源）。
2. **4D 毫米波雷达融合**：高速运动目标检测、**直接速度感知**（替代/补充里程计）、尘雨雾恶劣环境可靠性（电磁波穿透优于光学）。

## 与本库的关联

- 视觉 SLAM 侧：[[wiki/entities/hardware/UMI]]（ORB-SLAM3 + GoPro 鱼眼野外采集）。
- 状态估计 / 滤波：与 [[wiki/concepts/rl/RECAP]] 的贝叶斯反向消除、[[wiki/entities/systems/HIL-SERL]] 等共享 EKF/概率推断基础。
- 核心概念页：[[wiki/concepts/slam/SLAM (Simultaneous Localization and Mapping)]]（含严谨数学推导）。
- 互补视角：本文是 **被动（passive）SLAM**——给定运动轨迹做估计；而 [[wiki/sources/slam/2026-06-07 Active SLAM Survey (Placed 2023)]] 研究 **主动控制机器人运动以最小化不确定性**。
