---
type: entity
tags: [Data Collection, Hand-Held Gripper, Hardware, Imitation Learning, In-the-Wild, Open Source]
sources: [Chi 等 - 2024 - Universal Manipulation Interface, Zhaxizhuoma 等 - 2025 - FastUMI, Guo 等 - 2025 - DemoSpeedup]
created: 2026-04-24
updated: 2026-04-24
---

# UMI（Universal Manipulation Interface）

> **一句话**：手持 3D 打印平行夹爪 + GoPro 鱼眼镜头的低成本野外数据采集框架，通过精心设计的策略接口（延迟匹配 + 相对轨迹）实现人类示教到多机器人平台的零样本迁移。

**来源**：[[wiki/sources/data-collection/2026-04-24 UMI (Chi 2024)]]（原始论文，RSS 2024）  
**衍生**：[[wiki/sources/data-collection/2026-04-24 FastUMI (Zhaxizhuoma 2025)]]（硬件解耦重设计）  
**使用**：[[wiki/sources/data-efficiency/2026-04-23 DemoSpeedup]]（基于 UMI 数据的演示加速研究）  
**代码 / 硬件**：https://umi-gripper.github.io

---

## 系统组成

### 硬件

| 组件      | 规格                                                |
| ------- | ------------------------------------------------- |
| 结构      | 3D 打印平行夹爪，软指，触发激活                                 |
| 相机      | GoPro + 155° 鱼眼镜头（唯一传感器）                          |
| 追踪      | GoPro 内置 IMU + ORB-SLAM3（inertial-monocular SLAM） |
| 立体视觉    | 两块物理侧面镜子 + 数字反射处理                                 |
| 夹爪感知    | ArUco 标记追踪连续夹爪宽度                                  |
| 重量 / 尺寸 | 780 g；310 × 175 × 210 mm；80 mm 指尖行程               |
| 成本      | $73（3D 打印）+ $298（GoPro）= **$371 总计**              |

### 关键策略接口

1. **推理时延迟匹配（Inference-time Latency Matching）**：分别补偿观测延迟和执行延迟，保证训练/推理数据分布一致
2. **相对轨迹动作表示（Relative Trajectory Action Representation）**：所有 EE 位姿以当前时刻为参考，无需绝对坐标标定，天然支持跨机器人迁移
3. **[[wiki/concepts/generative-models/Diffusion Policy]] 作为策略主干**：建模多模态动作分布

---

## 核心性能（原始论文）

| 任务 | 成功率 |
|------|--------|
| Cup Arrangement（窄域）| 100% |
| Dynamic Tossing | 87.5% |
| Bimanual Cloth Folding | 70% |
| Dish Washing（7 步长视野）| 70% |
| Cup Arrangement（野外泛化）| **71.7%**（0-shot 跨环境）|
| Cross-robot（Franka FR2）| 90% |

野外泛化实验：12 人时 × 30 地点 × 1,400 条演示，零样本迁移到未见环境和对象。

---

## 生态与延伸

### FastUMI（Zhaxizhuoma 等, 2025）

[[wiki/entities/hardware/UMI]] 的全面工程重设计，解决了两大限制：

| 问题 | UMI | FastUMI |
|------|-----|---------|
| 硬件耦合 | 强依赖 Weiss WSG-50 | 标准化插拔指尖（覆盖 Open X-Embodiment >90% 夹爪）|
| VIO 鲁棒性 | GoPro + ORB-SLAM3（遮挡时失效）| RealSense T265 + Loop Closure |
| 算法适配 | 原版 DP | Smooth-ACT / PoseACT / Depth-Enhanced DP |
| 开源数据 | 无 | **10,000 条 / 22 任务**开源数据集 |

### DemoSpeedup（Guo 等, 2025）

基于 UMI 采集的演示数据，使用条件动作熵 + HDBSCAN 聚类实现 1.7×–3× 演示加速（详见 [[wiki/sources/data-efficiency/2026-04-23 DemoSpeedup]]）。

---

## Related Concepts

- [[wiki/concepts/generative-models/Diffusion Policy]]：UMI 策略主干
- [[wiki/concepts/imitation-learning/ACT]]：UMI 论文中提及的可替代策略框架；FastUMI 的对比基准之一
