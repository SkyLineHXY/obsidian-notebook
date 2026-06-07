---
type: concept
tags: [SLAM, LiDAR, IMU, KalmanFilter, StateEstimation, LiDAR-Inertial Odometry, iEKF, Manifold]
sources: [wiki/sources/slam/2026-06-07 FAST-LIO (Xu 2021), wiki/sources/slam/2026-06-07 FAST-LIO2 (Xu 2022)]
created: 2026-06-07
updated: 2026-06-07
---

# LiDAR-Inertial Odometry（LIO）

## 定义

LiDAR-Inertial Odometry（LIO）是同步融合 **LiDAR 点云**与 **IMU 惯性测量**来估计机器人 6-DOF 状态（位置 + 姿态 + 速度）的技术。LiDAR 提供精确的 3D 几何测量，IMU 提供高频（100–1000 Hz）运动预测，二者互补：LiDAR 修正 IMU 漂移，IMU 补偿 LiDAR 运动畸变。

**所在知识库来源**：
- [[wiki/sources/slam/2026-06-07 FAST-LIO (Xu 2021)]] — 紧耦合 iEKF + 高效 Kalman 增益
- [[wiki/sources/slam/2026-06-07 FAST-LIO2 (Xu 2022)]] — 直接点配准 + ikd-Tree 增量建图

---

## 核心挑战

1. **运动畸变（Motion Distortion）**：LiDAR 顺序扫描，一帧内各点时刻不同，高速运动时帧内位移不可忽略。
2. **计算效率**：一帧数千点的配准 + 融合需在 10–100 ms 内完成。
3. **退化（Degeneration）**：结构单一环境（走廊、空旷室外）沿某些方向几何约束不足。
4. **外参标定**：LiDAR-IMU 相对位姿 ${}^I\mathbf{T}_L$ 影响精度，需在线估计。

---

## 松耦合 vs 紧耦合

| 方式 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **松耦合** | 先点云配准得位姿，再融合 IMU | 模块独立，实现简单 | 忽略状态相关性；退化时 scan reg 不可靠 |
| **紧耦合** | 原始特征/点直接与 IMU 联合滤波 | 退化鲁棒；IMU 可约束退化方向 | 实现复杂；计算量大 |

FAST-LIO / FAST-LIO2 均采用**紧耦合**方案。

---

## 严谨数学推导

### 1. 符号定义

| 符号 | 含义 |
|------|------|
| $G$ | 全局坐标系（第一帧 IMU 系） |
| $I$ | IMU 体坐标系 |
| $L$ | LiDAR 体坐标系 |
| ${}^G\mathbf{R}_I \in SO(3)$ | IMU 在全局系的姿态矩阵 |
| ${}^G\mathbf{p}_I, {}^G\mathbf{v}_I \in \mathbb{R}^3$ | IMU 位置、速度 |
| $\mathbf{b}_\omega, \mathbf{b}_\mathbf{a} \in \mathbb{R}^3$ | 陀螺仪/加速度计偏置 |
| ${}^G\mathbf{g} \in \mathbb{R}^3$ | 重力向量 |
| $\omega_m, \mathbf{a}_m$ | IMU 原始测量 |
| $\mathbf{n}_\omega, \mathbf{n}_\mathbf{a}$ | IMU 测量白噪声 |
| $\mathbf{n}_{b\omega}, \mathbf{n}_{ba}$ | 偏置随机游走噪声 |
| ${}^I\mathbf{T}_L = ({}^I\mathbf{R}_L, {}^I\mathbf{p}_L)$ | LiDAR-IMU 外参 |

### 2. 流形运算 $\boxplus$ / $\boxminus$

状态空间是复合流形 $\mathcal{M}$，需要流形加法：

$$
\boxplus: \mathcal{M} \times \mathbb{R}^n \to \mathcal{M}; \qquad \boxminus: \mathcal{M} \times \mathcal{M} \to \mathbb{R}^n
$$

在 $SO(3)$ 上（以右乘为约定）：

$$
\mathbf{R} \boxplus \mathbf{r} = \mathbf{R}\,\mathrm{Exp}(\mathbf{r}), \qquad \mathbf{R}_1 \boxminus \mathbf{R}_2 = \mathrm{Log}(\mathbf{R}_2^T \mathbf{R}_1)
$$

其中指数映射 $\mathrm{Exp}: \mathbb{R}^3 \to SO(3)$（Rodrigues 公式）：

$$
\mathrm{Exp}(\mathbf{r}) = \mathbf{I} + \frac{\mathbf{r}}{\|\mathbf{r}\|}\sin(\|\mathbf{r}\|) + \frac{\mathbf{r}^2}{\|\mathbf{r}\|^2}(1 - \cos(\|\mathbf{r}\|))
$$

在 $\mathbb{R}^n$ 上退化为普通加减法。满足恒等式：

$$
(\mathbf{x} \boxplus \mathbf{u}) \boxminus \mathbf{x} = \mathbf{u}, \qquad \mathbf{x} \boxplus (\mathbf{y} \boxminus \mathbf{x}) = \mathbf{y}
$$

### 3. IMU 连续运动模型

$$
\begin{aligned}
{}^G\dot{\mathbf{R}}_I &= {}^G\mathbf{R}_I \lfloor \omega_m - \mathbf{b}_\omega - \mathbf{n}_\omega \rfloor_\wedge \\
{}^G\dot{\mathbf{p}}_I &= {}^G\mathbf{v}_I \\
{}^G\dot{\mathbf{v}}_I &= {}^G\mathbf{R}_I(\mathbf{a}_m - \mathbf{b}_\mathbf{a} - \mathbf{n}_\mathbf{a}) + {}^G\mathbf{g} \\
\dot{\mathbf{b}}_\omega &= \mathbf{n}_{b\omega}, \quad \dot{\mathbf{b}}_\mathbf{a} = \mathbf{n}_{ba}, \quad {}^G\dot{\mathbf{g}} = \mathbf{0}
\end{aligned}
$$

其中 $\lfloor \mathbf{a} \rfloor_\wedge$ 为向量 $\mathbf{a}$ 对应的反对称矩阵（叉积运算）。

### 4. 离散化状态转移方程

以 IMU 采样间隔 $\Delta t$ 零阶保持离散化：

$$
\mathbf{x}_{i+1} = \mathbf{x}_i \boxplus (\Delta t\, \mathbf{f}(\mathbf{x}_i, \mathbf{u}_i, \mathbf{w}_i)) \tag{STM}
$$

其中 $\mathbf{u}_i = [\omega_m^T, \mathbf{a}_m^T]^T$，$\mathbf{w}_i = [\mathbf{n}_\omega^T, \mathbf{n}_\mathbf{a}^T, \mathbf{n}_{b\omega}^T, \mathbf{n}_{ba}^T]^T$。

**误差状态线性化**（在名义状态 $\widehat{\mathbf{x}}_i$ 处展开，设 $\widetilde{\mathbf{x}}_i = \mathbf{x}_i \boxminus \widehat{\mathbf{x}}_i$）：

$$
\widetilde{\mathbf{x}}_{i+1} \simeq \mathbf{F}_{\widetilde{\mathbf{x}}} \widetilde{\mathbf{x}}_i + \mathbf{F}_\mathbf{w} \mathbf{w}_i
$$

误差状态转移矩阵（$18 \times 18$，FAST-LIO 版）：

$$
\mathbf{F}_{\widetilde{\mathbf{x}}} = \begin{bmatrix}
\mathrm{Exp}(-\widehat{\omega}_i\Delta t) & \mathbf{0} & \mathbf{0} & -\mathbf{A}(\widehat{\omega}_i\Delta t)^T\Delta t & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{I} & \mathbf{I}\Delta t & \mathbf{0} & \mathbf{0} & \mathbf{0} \\
-{}^G\widehat{\mathbf{R}}_{I_i}\lfloor\widehat{\mathbf{a}}_i\rfloor_\wedge\Delta t & \mathbf{0} & \mathbf{I} & \mathbf{0} & -{}^G\widehat{\mathbf{R}}_{I_i}\Delta t & \mathbf{I}\Delta t \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I} & \mathbf{0} \\
\mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{0} & \mathbf{I}
\end{bmatrix}
$$

协方差前向传播（对每个 IMU 测量执行一次）：

$$
\widehat{\mathbf{P}}_{i+1} = \mathbf{F}_{\widetilde{\mathbf{x}}} \widehat{\mathbf{P}}_i \mathbf{F}_{\widetilde{\mathbf{x}}}^T + \mathbf{F}_\mathbf{w} \mathbf{Q} \mathbf{F}_\mathbf{w}^T
$$

### 5. 运动畸变补偿（后向传播）

设帧尾时刻 $t_k$，特征点 $j$ 在时刻 $\rho_j \leq t_k$ 采样。从 $t_k$ 向前回溯至 $\rho_j$：

$$
{}^{I_k}\check{\mathbf{R}}_{I_{j-1}} = {}^{I_k}\check{\mathbf{R}}_{I_j} \cdot \mathrm{Exp}\left((\widehat{\mathbf{b}}_{\omega_k} - \omega_{m_{i-1}})\Delta t\right)
$$

$$
{}^{I_k}\check{\mathbf{p}}_{I_{j-1}} = {}^{I_k}\check{\mathbf{p}}_{I_j} - {}^{I_k}\check{\mathbf{v}}_{I_j}\Delta t
$$

得到相对变换 ${}^{I_k}\check{\mathbf{T}}_{I_j}$，将点投影到帧尾 LiDAR 坐标系：

$$
{}^{L_k}\mathbf{p}_{f_j} = {}^I\mathbf{T}_L^{-1} \cdot {}^{I_k}\check{\mathbf{T}}_{I_j} \cdot {}^I\mathbf{T}_L \cdot {}^{L_j}\mathbf{p}_{f_j}
$$

### 6. 测量模型（点到平面残差）

对 FAST-LIO（特征点）：残差为点到最近平面/边缘的距离：

$$
\mathbf{z}_j^\kappa = \mathbf{G}_j\left({}^G\widehat{\mathbf{p}}_{f_j}^\kappa - {}^G\mathbf{q}_j\right), \quad \mathbf{G}_j = \mathbf{u}_j^T \; (\text{平面法向量})
$$

对 FAST-LIO2（直接点）：每个原始点假设落在 5 近邻构成的局部平面上：

$$
\mathbf{0} = \mathbf{h}_j(\mathbf{x}_k, {}^L\mathbf{n}_j) = {}^G\mathbf{u}_j^T\left({}^G\mathbf{T}_{I_k} \cdot {}^I\mathbf{T}_L\left({}^L\mathbf{p}_j + {}^L\mathbf{n}_j\right) - {}^G\mathbf{q}_j\right)
$$

线性化后测量雅可比 $\mathbf{H}_j^\kappa$（对误差状态 $\widetilde{\mathbf{x}}_k^\kappa$ 求偏导）。

### 7. 迭代 EKF 更新（iEKF on Manifold）

**MAP 优化目标**（将先验 + 所有点测量联合）：

$$
\min_{\widetilde{\mathbf{x}}_k^\kappa} \underbrace{\|\mathbf{x}_k \boxminus \widehat{\mathbf{x}}_k\|^2_{\widehat{\mathbf{P}}_k^{-1}}}_{\text{先验项}} + \underbrace{\sum_{j=1}^m \|\mathbf{z}_j^\kappa + \mathbf{H}_j^\kappa \widetilde{\mathbf{x}}_k^\kappa\|^2_{\mathbf{R}_j^{-1}}}_{\text{测量项}}
$$

先验在当前迭代点处的线性化（$\mathbf{J}^\kappa$ 为流形切空间的 Jacobian）：

$$
\mathbf{x}_k \boxminus \widehat{\mathbf{x}}_k = \widehat{\mathbf{x}}_k^\kappa \boxminus \widehat{\mathbf{x}}_k + \mathbf{J}^\kappa \widetilde{\mathbf{x}}_k^\kappa, \quad \mathbf{J}^\kappa = \begin{bmatrix}\mathbf{A}(\delta{}^G\theta_{I_k})^{-T} & \mathbf{0} \\ \mathbf{0} & \mathbf{I}\end{bmatrix}
$$

令 $\mathbf{P} = (\mathbf{J}^\kappa)^{-1}\widehat{\mathbf{P}}_k(\mathbf{J}^\kappa)^{-T}$，目标函数变为标准二次型，其最优解（Kalman 更新步）：

$$
\widehat{\mathbf{x}}_k^{\kappa+1} = \widehat{\mathbf{x}}_k^\kappa \boxplus \left(-\mathbf{K}\mathbf{z}_k^\kappa - (\mathbf{I} - \mathbf{K}\mathbf{H})(\mathbf{J}^\kappa)^{-1}(\widehat{\mathbf{x}}_k^\kappa \boxminus \widehat{\mathbf{x}}_k)\right)
$$

### 8. 高效 Kalman 增益（FAST-LIO 核心定理）

**两种等价形式**（由矩阵求逆引理证明）：

**传统形式**（测量维度 $m$ 的矩阵求逆）：
$$
\mathbf{K}_\text{trad} = \mathbf{P}\mathbf{H}^T(\mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R})^{-1} \quad \leftarrow O(m^2), \; m \sim 10^3
$$

**新形式**（状态维度 $n$ 的矩阵求逆）：
$$
\mathbf{K}_\text{new} = (\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H} + \mathbf{P}^{-1})^{-1}\mathbf{H}^T\mathbf{R}^{-1} \quad \leftarrow O(n^2), \; n \in \{18, 24\}
$$

**等价性证明**：由矩阵求逆引理 $(\mathbf{P}^{-1} + \mathbf{H}^T\mathbf{R}^{-1}\mathbf{H})^{-1} = \mathbf{P} - \mathbf{P}\mathbf{H}^T(\mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R})^{-1}\mathbf{H}\mathbf{P}$，代入新形式展开可得传统形式。$\blacksquare$

由于 $\mathbf{R}$ 为块对角矩阵（各点测量独立），$\mathbf{H}^T\mathbf{R}^{-1}\mathbf{H} = \sum_j \mathbf{H}_j^T \mathbf{R}_j^{-1} \mathbf{H}_j$（$n \times n$），可增量累积，无需存储整个 $\mathbf{H}$ 矩阵。

### 9. 收敛后最优估计

$$
\bar{\mathbf{x}}_k = \widehat{\mathbf{x}}_k^{\kappa+1}, \quad \bar{\mathbf{P}}_k = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}
$$

---

## 典型系统对比

| 系统 | 耦合方式 | 点云处理 | 地图结构 | 外参 | 频率 |
|------|----------|----------|----------|------|------|
| LOAM | 松 | 特征（边/面） | 全量 KD-tree | 固定 | 2 Hz mapping |
| LINS | 紧（iEKF） | 特征（边/面） | 小局部地图 | 固定 | 10 Hz |
| FAST-LIO | 紧（iEKF） | 特征（边/面） | 全量 KD-tree | 固定 | 10–50 Hz |
| FAST-LIO2 | 紧（iEKF） | 直接原始点 | ikd-Tree 增量 | 在线标定 | 10–100 Hz |

---

## 关联页面

- [[wiki/sources/slam/2026-06-07 FAST-LIO (Xu 2021)]]
- [[wiki/sources/slam/2026-06-07 FAST-LIO2 (Xu 2022)]]
- [[wiki/concepts/slam/SLAM (Simultaneous Localization and Mapping)]]
- [[wiki/sources/slam/2026-06-07 GPS-Denied LiDAR-Based SLAM Survey (Jiang 2025)]]
