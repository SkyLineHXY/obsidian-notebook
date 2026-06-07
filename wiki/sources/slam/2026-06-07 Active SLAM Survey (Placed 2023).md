---
type: source
tags: [SLAM, ActiveSLAM, Survey, POMDP, InformationTheory, DRL, Robotics, Exploration]
sources: [raw/sources/papers/其他/Placed 等 - 2023 - A Survey on Active Simultaneous Localization and Mapping State of the Art and New Frontiers/Placed 等 - 2023 - A Survey on Active Simultaneous Localization and Mapping State of the Art and New Frontiers.md]
created: 2026-06-07
updated: 2026-06-07
---

# A Survey on Active SLAM: State of the Art and New Frontiers (Placed et al. 2023)

> 原文回链：[[raw/sources/papers/其他/Placed 等 - 2023 - A Survey on Active Simultaneous Localization and Mapping State of the Art and New Frontiers/Placed 等 - 2023 - A Survey on Active Simultaneous Localization and Mapping State of the Art and New Frontiers.md]]
> 出处：*IEEE Transactions on Robotics (T-RO)*, vol. 39, 2023, DOI 10.1109/TRO.2023.3248510。
> 作者：Julio A. Placed, Jared Strader, Henry Carrillo, Nikolay Atanasov, Vadim Indelman, Luca Carlone, José A. Castellanos（Zaragoza / MIT / UCSD / Technion）。

## 一句话总结

Active SLAM 领域 **里程碑式综述**（T-RO）：首次给出 **POMDP 统一问题表述**，系统梳理经典 **三阶段模块化方案**（目标识别 → 效用计算 → 动作选择），再覆盖 **置信空间规划（BSP）/ 连续优化 / 深度强化学习 / 多机器人** 等替代路线，最后列出 7 大开放挑战。是从被动 SLAM 走向 **主动决策** 的权威 roadmap。

## 核心定义

> **Active SLAM** = 规划并控制机器人运动，以构建最准确、最完整的环境模型。本质是 **探索–利用（exploration–exploitation）困境** 下的决策问题：探索新区域 vs 重访旧区域（触发回环以提升定位精度）。

- 历史沿革：active perception（Bajcsy 1985）→ next-best-view（active mapping, Connolly 1985）→ active localization（Fox 1998，奠定"目标识别/效用计算/动作选择"三段式）→ active SLAM（Davison & Murray 2002 正式命名）。
- 出版量从 2010 年 53 篇增至 2022 年 660+ 篇（12 倍增长），但仅 31 项专利 vs 普通 SLAM 的 ~39000 项——技术成熟度仍处早期。

## 1. 统一问题表述（POMDP）

- 7 元组 POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{Z}, \xi_s, \xi_z, r, \gamma)$；状态空间为联合 $\mathcal{S} \triangleq \mathcal{X} \times \mathcal{M}$（位姿 × 地图）。
- 维护 **信念（belief）** $b_t(\boldsymbol{s}_t) \triangleq p(\boldsymbol{s}_t \mid \boldsymbol{z}_{1:t}, \boldsymbol{a}_{1:t-1})$；最优策略 $\pi^\star(b) = \arg\max_\pi \sum_{t=0}^\infty \mathbb{E}[\gamma^t \rho(b_t, \pi(b_t))]$。
- 关键：active SLAM 的奖励是 **效用函数（utility）**，衡量 **信念的不确定性** 而非到达特定状态——这违背标准 POMDP（奖励依赖 $s,a$），需用 **ρ-POMDP** 扩展以容纳信息论目标。
- 有限时域 + ML 观测下退化为开环规划：$\boldsymbol{a}_{t:t+k}^\star = \arg\max \sum_{\tau} \rho(b(\boldsymbol{s}_\tau), \boldsymbol{a}_\tau)$。

## 2. 模块化三阶段方案

### Stage 1 — 目标识别（候选动作）
- 环境表示四类：拓扑图 / **度量图（metric，最常用：稀疏 landmark + 稠密 OG/OctoMap/voxel）** / 度量-语义图（Kimera、3D scene graph）/ 混合分层图。
- **Frontier（前沿）** 是主流（Yamauchi 1997）：已知与未知区域的边界。检测器：WFD/FFD、incremental WFD、RRT-based、sample-based。3D 前沿较少，需聚类（K-means/mean-shift）。
- 关键洞察：高不确定性时 **潜在回环区域比前沿更具信息量**（Stachniss/Newman）→ 候选集常同时含前沿 + 重访点，显式编码探索–利用困境。

### Stage 2 — 效用计算（核心）
- **朴素代价**：欧氏距离 / 时间 / 待访面积（≈地图熵）。
- **信息论（IT，最常用）**：SLAM 后验联合熵（式 10）= 机器人熵 + 期望条件地图熵；常近似为两项独立相加（需加权平衡量纲）。**互信息 MI** = 期望熵减（信息增益）；还有 KLD。
- **最优实验设计（TOED）**：直接量化任务空间协方差。Kiefer 族 $\|\boldsymbol{\Sigma}\|_p$ 给出四大 **最优性准则**——T-opt（迹/均值方差）、**D-opt（行列式/超椭球体积）**、A-opt（调和均值）、E-opt（最大/最小特征值）。
- **单调性定理**：探索时不确定性应单调增；[153] 证明仅 **D-opt** 保证单调，但 Kim/Rodríguez-Arévalo 指出单调性更取决于 **误差表示方式**（差分表示对所有准则保证单调）。
- **图结构等价**：分析 pose-graph 的拉普拉斯/代数连通度 ≡ 计算最优性准则（Khosoussi、Placed & Castellanos 在 SE(n) 上的工作），大幅降低算力。

### Stage 3 — 动作选择与执行
- 离散候选：枚举求 $\boldsymbol{a}^\star = \arg\min_{\boldsymbol{a}} \|\boldsymbol{\Sigma}\|_p = \arg\max_{\boldsymbol{a}} \|\boldsymbol{\Phi}\|_q$（FIM）或 $\arg\max \mathcal{I}_G$。
- 用 RRT/PRM/RRT* 导航到目标——但解耦"先选目标再规划路径"导致 **次优性**。

## 3. 替代方案

- **BSP / 连续优化**：直接优化未来轨迹（不离散化动作空间）。BRM（belief road map）、iLQG、值迭代推广到连续 POMDP。
- **最优控制**：线性高斯 + MI 奖励时退化为 **确定性** 最优控制（协方差独立于观测实现）→ 可用 LQG / DDP / MPC。
- **DRL**（Section VII）：绕过三段式，效用与选择嵌入网络。
  - 奖励设计：外在奖励（实为避障）→ **内在好奇心**（curiosity-driven）→ **不确定性感知**（引入 D-opt / T-opt / MI 作为奖励，更稳健）。
  - 难点：**部分可观**（用 LSTM/RNN 编码历史）、**泛化**（稀疏 range 输入降 sim-2-real gap）、**训练环境**（Stage/Gazebo/Habitat/iGibson）。
  - 趋势：planning + learning 结合（DRL 从已检测前沿中选最优）。
- **多机器人**：联合状态/动作空间集中式（随 n 指数爆炸）→ 去中心化/分布式（coordinate descent、连通度索引、Voronoi 分区 + DRL 安全导航）。

## 4. 七大开放挑战

1. **超视距预测（beyond line-of-sight）**：用 AE/VAE/SDF 预测未见区域地图与位姿不确定性；回环出现对新状态不确定性影响巨大却难预测。
2. **从 active SLAM 到 active spatial perception**：基于 3D scene graph 的语义/分层主动感知。
3. **鲁棒在线 BSP**：模糊环境下的 **数据关联**（假设消歧）。
4. **动态与可变形场景** 推理（目前几乎空白）。
5. **有意义的自主停止准则**：何时判定任务完成（TOED 度量是 promising 工具）。
6. **可复现研究**：active SLAM 因 agent 需与环境交互，无法用静态数据集 benchmark，缺乏统一评测。
7. **实际应用**：水下船体检测、矿井/地下、火星 analogue 等少量落地；算力高 vs 机器人预算低的矛盾。

## 与本库的关联

- 互补视角：本文 = **主动 SLAM**（控制运动以降不确定性）；[[wiki/sources/slam/2026-06-07 GPS-Denied LiDAR-Based SLAM Survey (Jiang 2025)]] = **被动 LiDAR SLAM**（给定轨迹做估计）。两者构成 SLAM 完整图景。
- 方法论交叉：POMDP / belief / 策略优化与 [[wiki/concepts/rl/Offline 强化学习]]、[[wiki/concepts/rl/DPPO]] 共享 RL 决策框架；DRL active SLAM 与本库 VLA+RL 族同源。
- 信息论效用（熵 / MI / Fisher 信息）与 [[wiki/analyses/DemoSpeedup 逐帧熵估计与HDBSCAN聚类数学推导]] 的熵估计互通。
- 核心概念页：[[wiki/concepts/slam/SLAM (Simultaneous Localization and Mapping)]]（含 active SLAM 的 POMDP 与最优性准则严谨推导）。
