---
type: concept
tags: [SLAM, StateEstimation, KalmanFilter, POMDP, InformationTheory, Robotics, ActiveSLAM]
sources: [raw/sources/papers/其他/Jiang 等 - 2025 - GPS-Denied LiDAR-Based SLAM—A Survey/Jiang 等 - 2025 - GPS-Denied LiDAR-Based SLAM—A Survey.md, raw/sources/papers/其他/Placed 等 - 2023 - A Survey on Active Simultaneous Localization and Mapping State of the Art and New Frontiers/Placed 等 - 2023 - A Survey on Active Simultaneous Localization and Mapping State of the Art and New Frontiers.md, raw/assets/papers/其他/Tuna 等 - 2024 - X-ICP Localizability-Aware LiDAR Registration for Robust Localization in Extreme Environments.pdf]
created: 2026-06-07
updated: 2026-06-07
---

# SLAM (Simultaneous Localization and Mapping)

**SLAM（同步定位与建图）** 指机器人在 **未知环境** 中，一边增量构建环境地图、一边在该地图中定位自身的问题。它是自主移动机器人的核心使能技术——在 GPS 拒止（室内/地下/隧道）场景尤为关键。

> 来源：综合 [[wiki/sources/slam/2026-06-07 GPS-Denied LiDAR-Based SLAM Survey (Jiang 2025)]]（被动 LiDAR SLAM 流程与状态估计）与 [[wiki/sources/slam/2026-06-07 Active SLAM Survey (Placed 2023)]]（主动 SLAM 决策框架）。

## 1. 问题分解

经典 SLAM 系统分 **前端（front-end）** 与 **后端（back-end）**：
- **前端**：原始传感器数据处理——特征提取、scan matching（ICP / NDT）、帧间运动估计。
- **后端**：非线性优化降累积误差、回环检测（loop closure）保证全局一致性。

按 **传感器** 分：视觉 SLAM（ORB-SLAM、VINS-Mono；依赖光照/纹理）、LiDAR SLAM（LOAM、FAST-LIO；对光照鲁棒）、多传感器融合（LVI-SAM、FAST-LIVO）。

按 **后端更新方案** 分：
- **滤波式（filter-based）**：基于概率滤波递归估计状态。EKF / UKF / 粒子滤波（PF）。实时性强，工业首选。
- **优化式（optimization-based）**：转化为目标函数最小化（重投影误差、scan 匹配误差），用 BA / 图优化求解。精度高、算力大。

按 **是否控制运动** 分：
- **被动 SLAM（passive）**：运动轨迹给定（人遥控或外部规划），SLAM 仅做估计。
- **主动 SLAM（active）**：机器人 **自主决策运动** 以最小化定位与地图不确定性，是探索–利用困境下的 POMDP 决策问题。

## 2. 关键模块

| 模块    | 核心内容             | 代表方法                                |
| ----- | ---------------- | ----------------------------------- |
| 前向传播  | IMU 高频递推位姿（帧间预测） | FAST-LIO 预积分                        |
| 后向传播  | LiDAR 残差修正漂移     | ICP / scan-to-model + EKF           |
| 关键帧选择 | 平衡存储与精度          | 信息增益（Fisher 信息）/ 阈值 / 运动            |
| 数据结构  | 点云高效组织           | KD-tree → IKD-tree → Octree → voxel |
| 回环检测  | 修正累积误差           | 图优化 / BoW3D / 深度描述子                 |
| 渲染    | 地图可视化            | 点云 / 彩色点云 / NeRF / 3DGS             |

## 3. 严谨数学推导

本节涵盖三个层次：**(A)** 概率 SLAM 后验的贝叶斯分解；**(B)** EKF-SLAM 的预测-更新递归（被动 SLAM 后端）；**(C)** 主动 SLAM 的 POMDP 表述与最优性准则（active SLAM 决策）。

### 3.1 符号定义

- $\boldsymbol{x}_{t} \in \mathrm{SE}(3)$：$t$ 时刻机器人位姿（旋转 $\boldsymbol{R}_t$ + 平移 $\boldsymbol{p}_t$）。
- $\boldsymbol{m} \in \mathcal{M}$：环境地图（landmark 集合或占据栅格）。
- $\boldsymbol{z}_{1:t}$：观测序列；$\boldsymbol{a}_{1:t-1}$（或 $\boldsymbol{u}_{1:t}$）：控制/里程输入序列。
- $b_t(\boldsymbol{s}_t) = p(\boldsymbol{s}_t \mid \boldsymbol{z}_{1:t}, \boldsymbol{a}_{1:t-1})$：信念（belief），联合状态 $\boldsymbol{s}=(\boldsymbol{x},\boldsymbol{m})$ 的后验。
- $\boldsymbol{\Sigma}$：状态后验协方差矩阵；$\boldsymbol{\Phi}=\boldsymbol{\Sigma}^{-1}$：Fisher 信息矩阵（FIM）。

### 3.2 (A) 概率 SLAM 后验

**目标**：求联合后验 $p(\boldsymbol{x}_{0:t}, \boldsymbol{m} \mid \boldsymbol{z}_{1:t}, \boldsymbol{u}_{1:t})$。

利用 **马尔可夫假设**（状态转移仅依赖前一状态与当前控制）和 **观测条件独立性**（给定状态，观测相互独立），由贝叶斯定理：

$$
p(\boldsymbol{x}_{0:t}, \boldsymbol{m} \mid \boldsymbol{z}_{1:t}, \boldsymbol{u}_{1:t}) \propto p(\boldsymbol{x}_0)\, \prod_{k=1}^{t} \underbrace{p(\boldsymbol{x}_k \mid \boldsymbol{x}_{k-1}, \boldsymbol{u}_k)}_{\text{运动模型}}\; \underbrace{p(\boldsymbol{z}_k \mid \boldsymbol{x}_k, \boldsymbol{m})}_{\text{观测模型}}.
$$

**推导步骤**：
1. 贝叶斯：$p(\boldsymbol{x}_{0:t},\boldsymbol{m}\mid \boldsymbol{z}_{1:t},\boldsymbol{u}_{1:t}) \propto p(\boldsymbol{z}_t \mid \boldsymbol{x}_{0:t},\boldsymbol{m},\boldsymbol{z}_{1:t-1},\boldsymbol{u}_{1:t})\, p(\boldsymbol{x}_{0:t},\boldsymbol{m}\mid \boldsymbol{z}_{1:t-1},\boldsymbol{u}_{1:t})$。
2. 观测独立性：$p(\boldsymbol{z}_t \mid \cdots) = p(\boldsymbol{z}_t \mid \boldsymbol{x}_t, \boldsymbol{m})$。
3. 链式展开 $\boldsymbol{x}_t$：$p(\boldsymbol{x}_t \mid \boldsymbol{x}_{0:t-1}, \cdots) = p(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{u}_t)$（马尔可夫）。
4. 对 $k=t,t-1,\dots,1$ 递归即得上式连乘。

**直觉**：SLAM 后验 = 先验 × 所有"运动预测"× 所有"观测修正"。这正是 **因子图（factor graph）** 的数学本质——每个因子是一个运动或观测约束，后端图优化即最大化此后验（等价最小化负对数 = 加权残差平方和）。

### 3.3 (B) EKF-SLAM 递归（被动后端）

将信念近似为高斯 $b_t = \mathcal{N}(\hat{\boldsymbol{s}}_t, \boldsymbol{\Sigma}_t)$，运动/观测模型一阶泰勒线性化。

**预测（前向传播，对应 Jiang 综述 §3.1）**：以运动模型 $\boldsymbol{s}_k = f(\boldsymbol{s}_{k-1}, \boldsymbol{u}_k) + \boldsymbol{w}_k$，$\boldsymbol{w}_k \sim \mathcal{N}(0, \boldsymbol{Q}_k)$：

$$
\hat{\boldsymbol{s}}_k^- = f(\hat{\boldsymbol{s}}_{k-1}, \boldsymbol{u}_k), \qquad
\boldsymbol{\Sigma}_k^- = \boldsymbol{F}_{k-1}\boldsymbol{\Sigma}_{k-1}\boldsymbol{F}_{k-1}^\mathrm{T} + \boldsymbol{G}_{k-1}\boldsymbol{Q}_{k-1}\boldsymbol{G}_{k-1}^\mathrm{T},
$$

其中 $\boldsymbol{F}_{k-1} = \left.\frac{\partial f}{\partial \boldsymbol{s}}\right|_{\hat{\boldsymbol{s}}_{k-1}}$ 为状态转移雅可比，$\boldsymbol{G}_{k-1}$ 为噪声雅可比。**协方差单调增长** ⇒ 仅靠预测必然漂移。

**更新（后向传播，对应残差修正）**：以观测模型 $\boldsymbol{z}_k = h(\boldsymbol{s}_k) + \boldsymbol{n}_k$，$\boldsymbol{n}_k \sim \mathcal{N}(0, \boldsymbol{R}_n)$，雅可比 $\boldsymbol{H}_k = \left.\frac{\partial h}{\partial \boldsymbol{s}}\right|_{\hat{\boldsymbol{s}}_k^-}$：

$$
\tilde{\boldsymbol{r}}_k = \boldsymbol{z}_k - h(\hat{\boldsymbol{s}}_k^-) \quad(\text{创新/残差}),
$$
$$
\boldsymbol{S}_k = \boldsymbol{H}_k \boldsymbol{\Sigma}_k^- \boldsymbol{H}_k^\mathrm{T} + \boldsymbol{R}_n, \qquad
\boldsymbol{K}_k = \boldsymbol{\Sigma}_k^- \boldsymbol{H}_k^\mathrm{T} \boldsymbol{S}_k^{-1} \quad(\text{Kalman 增益}),
$$
$$
\hat{\boldsymbol{s}}_k = \hat{\boldsymbol{s}}_k^- + \boldsymbol{K}_k \tilde{\boldsymbol{r}}_k, \qquad
\boldsymbol{\Sigma}_k = (\boldsymbol{I} - \boldsymbol{K}_k \boldsymbol{H}_k)\boldsymbol{\Sigma}_k^-.
$$

**关键步骤直觉**：
- **Kalman 增益** $\boldsymbol{K}_k$ 是"信任权重"——观测噪声 $\boldsymbol{R}_n$ 越小则越信任观测（$\boldsymbol{K}_k$ 越大），预测不确定性 $\boldsymbol{\Sigma}_k^-$ 越大则越依赖修正。
- **更新使协方差收缩**：$\boldsymbol{\Sigma}_k \preceq \boldsymbol{\Sigma}_k^-$（半正定意义下），这正是 LiDAR 点云残差对抗 IMU 漂移的数学体现。回环检测则提供"跨大时间跨度"的强约束，一次性大幅压缩 $\boldsymbol{\Sigma}$。
- LiDAR SLAM 中 $h(\cdot)$ 即点云配准（ICP），残差 $\tilde{\boldsymbol{r}}$ 是点对欧氏距离 $\boldsymbol{p}^\mathrm{o} - \boldsymbol{p}^\mathrm{p}$，最近邻用 KD-tree 加速。

### 3.4 (C) 主动 SLAM：POMDP 与最优性准则

被动 SLAM 到此为止；**主动 SLAM** 进一步 **选择控制 $\boldsymbol{a}$** 来最小化未来 $\boldsymbol{\Sigma}$。

**目标**：求最优策略 $\pi^\star$ 最大化期望累积效用：

$$
\pi^\star(b) = \arg\max_\pi \sum_{t=0}^\infty \mathbb{E}\big[\gamma^t \rho(b_t, \pi(b_t))\big], \qquad
\rho: \mathcal{B}(\mathcal{S}) \times \mathcal{A} \mapsto \mathbb{R}.
$$

效用 $\rho$ 衡量 **信念不确定性**（而非到达目标），需用 ρ-POMDP 框架容纳。有限时域 + 最大似然观测下退化为开环：

$$
\boldsymbol{a}_{t:t+k}^\star = \arg\max_{\boldsymbol{a}_{t:t+k}} \sum_{\tau=t}^{t+k} \rho(b(\boldsymbol{s}_\tau), \boldsymbol{a}_\tau).
$$

**效用的两条量化路线**：

**(i) 信息论（IT）**——后验联合熵（高斯下）：

$$
\mathcal{H}[p(\boldsymbol{x} \mid \cdot)] = \tfrac{1}{2}\ln\big((2\pi e)^{\ell} \det(\boldsymbol{\Sigma}_r)\big).
$$

互信息（信息增益）= 当前熵 − 期望后验熵：$\mathcal{I}(\boldsymbol{a}) = \mathcal{H}[p(\boldsymbol{s}\mid \boldsymbol{h})] - \mathbb{E}_{\hat{\boldsymbol{z}}}\big[\mathcal{H}[p(\boldsymbol{s}\mid \boldsymbol{h}, \hat{\boldsymbol{z}}, \boldsymbol{a})]\big]$。

**(ii) 最优实验设计（TOED）**——直接处理协方差 $\boldsymbol{\Sigma}$ 的特征值 $(\lambda_1,\dots,\lambda_n)$。Kiefer 族标量映射：

$$
\|\boldsymbol{\Sigma}\|_p \triangleq \Big(\tfrac{1}{n}\,\mathrm{trace}(\boldsymbol{\Sigma}^p)\Big)^{1/p}
= \begin{cases} \big(\tfrac{1}{n}\sum_k \lambda_k^p\big)^{1/p}, & 0<|p|<\infty,\\[4pt] \exp\big(\tfrac{1}{n}\sum_k \log\lambda_k\big), & p=0. \end{cases}
$$

边界情形给出四大 **最优性准则**：

| 准则 | $p$ | 表达式 | 几何含义 |
|---|---|---|---|
| T-opt | $1$ | $\frac1n\sum_k \lambda_k$ | 平均方差（迹） |
| **D-opt** | $0$ | $\exp(\frac1n\sum_k \log\lambda_k)$ | 协方差超椭球 **体积** |
| A-opt | $-1$ | $(\frac1n\sum_k \lambda_k^{-1})^{-1}$ | 调和均值方差 |
| E-opt | $\pm\infty$ | $\min/\max_k \lambda_k$ | 超椭球 **半径** |

**单调性定理（关键）**：探索时不确定性应单调增长，否则决策错误。文献证明在传统协方差表示下 **仅 D-opt 保证单调性**，故 D-opt 是 active SLAM 的"正确"效用函数；但后续工作（Rodríguez-Arévalo 等）证明：用 **差分（differential）误差表示** 时所有准则均保证单调——即 **单调性本质取决于不确定性的表示方式**，而非准则本身。

**直觉**：D-opt（行列式 = 特征值连乘）度量整个不确定性椭球的"体积"，对任一方向的高不确定性都敏感且乘性放大，因此最能捕捉"哪个动作最能压缩总体不确定性"。最优动作选择即：

$$
\boldsymbol{a}^\star = \arg\min_{\boldsymbol{a}} \|\boldsymbol{\Sigma}\|_p = \arg\max_{\boldsymbol{a}} \|\boldsymbol{\Phi}\|_q, \quad q=-p.
$$

**图论捷径**：由于 $\boldsymbol{\Phi}=\boldsymbol{\Sigma}^{-1}$ 稀疏，可证 pose-graph 的代数连通度 / 生成树数与 D-opt、E-opt 等价，从而以拉普拉斯谱替代昂贵的矩阵分解（Khosoussi、Placed & Castellanos 在 SE(n) 上的结果）。

## 4. 与本知识库的关联

- 被动 LiDAR SLAM 流程与实验：[[wiki/sources/slam/2026-06-07 GPS-Denied LiDAR-Based SLAM Survey (Jiang 2025)]]。
- 主动 SLAM 决策框架与开放挑战：[[wiki/sources/slam/2026-06-07 Active SLAM Survey (Placed 2023)]]。
- 数据采集中的 SLAM 应用：[[wiki/entities/hardware/UMI]]（ORB-SLAM3 手持采集）。
- 状态估计/概率推断方法论与 [[wiki/concepts/rl/RECAP]]（贝叶斯反向消除）、信息论效用与 [[wiki/analyses/DemoSpeedup 逐帧熵估计与HDBSCAN聚类数学推导]]（熵估计）相通。
- POMDP / belief / 策略优化决策框架与 [[wiki/concepts/rl/Offline 强化学习]]、[[wiki/concepts/rl/DPPO]] 同源。

## 5. 待扩展（单一来源，暂记 tag）

- **4D 毫米波雷达 SLAM 融合**（仅 Jiang 2025）：速度直接感知、恶劣环境穿透。
- **Active Spatial Perception / 3D Scene Graph 主动感知**（仅 Placed 2023）。
- **超视距地图预测（beyond line-of-sight）**：AE/VAE/SDF 场景补全（仅 Placed 2023）。
- **可变形/动态场景 active SLAM**（仅 Placed 2023）。
