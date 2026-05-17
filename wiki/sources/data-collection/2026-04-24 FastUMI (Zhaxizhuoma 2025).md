---
type: source
tags: [Imitation Learning, Data Collection, Hardware Decoupling, Diffusion Policy, ACT, Dataset, RealSense T265, In-the-Wild]
sources: [Zhaxizhuoma 等 - 2025 - FastUMI A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset.md]
created: 2026-04-24
updated: 2026-04-24
---

[[raw/sources/papers/IL(Imitation Learning)/Zhaxizhuoma 等 - 2025 - FastUMI A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset/Zhaxizhuoma 等 - 2025 - FastUMI A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset.md]]

# FastUMI (Zhaxizhuoma 等, 2025)

> **一句话**：对 [[wiki/entities/hardware/UMI]] 的全面工程重设计——解耦硬件、以 RealSense T265 替换 GoPro VIO、开源 10,000 条 22 任务演示数据集，实现真正的即插即用。

**论文**：Zhaxizhuoma 等, *"FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset"*, 2025  
**机构**：Shanghai AI Lab、上海交通大学、Bristol、复旦、HKU 等  
**项目**：https://fastumi.com

---

## 动机：UMI 的两大限制

1. **硬件强耦合**：原 UMI 依赖特定夹爪（Weiss WSG-50），跨平台部署需大量机械重设计、传感器重标定、代码修改，劳动成本高。
2. **VIO 管线脆弱**：基于 GoPro + ORB-SLAM3 的 VIO 在长时间遮挡（如铰链操作）时容易跟踪失败；参数敏感，部署门槛高。

---

## 系统重设计

### 硬件设计（三维解耦哲学）

| 解耦维度                                | 方案                                                                           |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| **物理解耦**（Physical Decoupling）       | 标准化插拔式指尖（Plug-in Fingertip），覆盖 Open X-Embodiment 数据集中 >90% 的夹爪型号；ISO 标准兼容法兰盘 |
| **视觉一致性**（Visual Consistency）       | 可调节相机安装结构（伸缩臂）保持手持设备与机器人端 GoPro 鱼眼视角一致，遵循"GoPro 底部与指尖底部对齐"标准规范               |
| **操作独立性**（Operational Independence） | T265 自包含 VIO 模块，消除外部标定与复杂参数调优                                                |

**手持设备组件**：
- **GoPro**（155° 鱼眼）：视觉观测 + 演示验证（不再负责位姿追踪）
- **RealSense T265**：位姿追踪（高性能内置 IMU，在部分遮挡下仍稳定）
- 指尖 + ArUco 标记：夹爪宽度追踪

**机器人端设备**：不含 T265；仅 GoPro + 插拔式指尖 + 可调安装结构。

### 软件框架

**数据采集**（ROS 三节点架构）：
- **相机节点**：1920×1080 @ 60 fps GoPro 视频流
- **追踪节点**：T265 位姿估计 @ 200 Hz，$(x, y, z, q_x, q_y, q_z, q_w)$ 四元数格式
- **存储节点**：聚合并同步多模态数据，写入 HDF5 文件

**多传感器同步**：统一 ROS 时钟 + 多线程缓冲，以最大公约频率（20 Hz）对 200 Hz T265 和 60 Hz GoPro 进行子采样，时间偏差亚毫秒级。

**T265 累积漂移修正**：
1. 静止复位（Reinitialization）：回到预定义参考姿态重置内部状态
2. 回环闭合（Loop Closure）：桌面放置蓝色 3D 打印凹槽作为视觉参考点，T265 重访时触发回环

---

## 轨迹推导（核心数学）

设 T265 在时刻 $i$ 输出相机位姿 $(\mathbf{p}_i, \mathbf{R}_i)$（相机局部坐标系），已知：相机到夹爪偏移 $\Delta_{c2g}$，夹爪中心在机器人基坐标系中的参考位姿 $(\mathbf{p}_{b2g}, \mathbf{R}_{b2g})$。

**绝对 TCP 轨迹**：

$$
\mathbf{p}_\text{cam}^{(i)} = \mathbf{p}_{b2g} + \mathbf{p}_i - \mathbf{R}_{b2g}\Delta_{c2g} \tag{1}
$$

$$
\mathbf{R}_\text{cam}^{(i)} = \mathbf{R}_\text{base} \cdot \mathbf{R}_i \tag{2}
$$

$$
\mathbf{p}_\text{ee}^{(i)} = \mathbf{p}_\text{cam}^{(i)} + \mathbf{R}_\text{cam}^{(i)}\Delta_{c2g}, \quad \mathbf{R}_\text{ee}^{(i)} = \mathbf{R}_\text{cam}^{(i)} \tag{3,4}
$$

**相对 TCP 轨迹**：

$$
\mathbf{p}_\text{rel}^{(i)} = \mathbf{p}_\text{ee}^{(i+1)} - \mathbf{p}_\text{ee}^{(i)} \tag{5}
$$

$$
\mathbf{R}_\text{rel}^{(i)} = \left(\mathbf{R}_\text{ee}^{(i)}\right)^{-1} \cdot \mathbf{R}_\text{ee}^{(i+1)} \tag{6}
$$

**夹爪宽度**（ArUco 标记像素距离 $d$ 映射）：

$$
W = \frac{d - d_\min}{d_\max - d_\min} \times G_\max \tag{7}
$$

---

## 算法适配

### Smooth-ACT：GRU 局部时序平滑

标准 [[wiki/concepts/imitation-learning/ACT]] 预测绝对关节轨迹，在第一人称视角下大量机械臂不可见，易产生运动学非法配置。Smooth-ACT 在 Transformer 解码器上叠加 GRU 层，损失函数：

$$
\mathcal{L} = \|\hat{a} - a\|_1 + \|\hat{a}_\text{GRU} - a\|_1 + \lambda \operatorname{KL}(\mu, \log\sigma^2) \tag{8}
$$

Transformer 捕获全局时空模式，GRU 约束局部连续性，联合减少运动学非法动作。

### PoseACT：末端执行器位姿预测

以 TCP 相对位姿替代绝对关节轨迹预测，降低对基坐标系和臂几何的敏感性，提升跨平台迁移。

### Depth-Enhanced DP

标准 [[wiki/concepts/generative-models/Diffusion Policy]]（Relative TCP 预测 + 延迟匹配）在精度敏感任务上因缺乏深度信息偶尔失败。FastUMI 方案：
- 用 **Depth Anything V2** 离线生成伪深度图（在 RTX 4090 上推理频率 20 Hz）
- 对 fisheye 圆形图像裁剪为内接矩形 $448 \times 448$ 后输入深度估计
- 深度图扩展为三通道伪彩色，与 RGB 分别通过 ViT-Base/16 CLIP 编码，拼接后用于 DP 下游

### 非平行夹爪动态误差补偿

非平行夹爪关闭时 TCP 沿本体 Z 轴位移，补偿算法：

$$
d(i) = d_\text{close} - \frac{d_\text{close} - d_\text{open}}{W_\max}W(i) \tag{9}
$$

$$
\mathbf{z}_\text{axis}^{(i)} = \mathbf{R}_\text{ee}^{(i)}\hat{\mathbf{e}}_z \tag{10}
$$

$$
\mathbf{p}_\text{ee}^{\prime(i)} = \mathbf{p}_\text{ee}^{(i)} - d(i)\mathbf{z}_\text{axis}^{(i)} \tag{11}
$$

$$
\boldsymbol{\theta}^{(i)} = \text{IK}\!\left(\mathbf{p}_\text{ee}^{\prime(i)}, \mathbf{R}_\text{ee}^{(i)}\right) \tag{12}
$$

---

## FastUMI 数据集

- **规模**：10,000 条演示轨迹（每条约 6–12 秒）
- **覆盖**：22 个日常任务、19 类物体、12 种操作技能
- **采集**：5 名操作者 × 3 台设备，家居场景，背景随机化
- **格式**：HDF5（含关节轨迹 + TCP 轨迹）+ Zarr 转换脚本
- **质量保障**：T265 置信度 ≥ High 的比例 ≥ 95%；低置信帧插值处理

---

## 实验结果

### 位姿追踪精度（vs MoCap 基准）

| 传感器            | Pick Cup（低遮挡） | Open Container（部分遮挡） | Rearrange Coke（重度遮挡） |
| -------------- | ------------- | -------------------- | -------------------- |
| RealSense T265 | **10.5 mm**   | 17.7 mm              | 23.3 mm              |
| RoboBaton MINI | 15.2 mm       | **11.2 mm**          | 稳定（更抗遮挡）             |

> T265 在无遮挡时精度最高；RoboBaton MINI 在遮挡变化场景中更鲁棒。

### 基准策略性能（12 任务，每任务 200 条演示 / 15 次测试）

| 操作类型 | DP（相对 TCP）平均 | ACT（绝对关节）平均 |
|---------|---------------|---------------|
| 铰链操作（开合）| 66.7% | 85.0% |
| 拾放操作 | 74.7% | 63.3% |
| 拾推操作 | 46.7% | 6.7% |
| 按钮操作 | 20.0% | 80.0% |

> ACT 在需要精确关节控制的刚性结构任务（铰链、按钮）上更优；DP 在需要灵活性的拾放和推拨任务上更优。

---

## Related Concepts & Entities

- [[wiki/entities/hardware/UMI]]：本文的直接前身与对比基准
- [[wiki/concepts/generative-models/Diffusion Policy]]：算法基准之一（Relative TCP 版本）
- [[wiki/concepts/imitation-learning/ACT]]：算法基准之一；本文提出 Smooth-ACT 和 PoseACT 变体
- [[wiki/sources/data-collection/2026-04-24 UMI (Chi 2024)]]：原始 UMI 论文
