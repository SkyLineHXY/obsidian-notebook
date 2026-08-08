---
type: analysis
tags: [UMI, Pose Transformation, SE3, Coordinate Frames, Imitation Learning, Inference, Robot Control, Calibration, Franka, Bimanual Manipulation, GoPro VIO, Hand-Eye Calibration]
sources: [Chi 等 - 2024 - Universal Manipulation Interface, Zhaxizhuoma 等 - 2025 - FastUMI]
created: 2026-04-25
updated: 2026-08-08 (新增 §10：双臂 GoPro VIO 的共同地图、Camera→TCP、双基座与 policy TCP 变换链)
---

# UMI ee6d 位姿变换：EE（法兰）-at-$t_0$ 推理完整推导

> **核心问题**：UMI 采集的 ee6d 数据如何以 $t_0$ 时刻**法兰坐标系**为参考表达，推理时机械臂如何将策略输出还原为可执行的基坐标系命令？

**关联来源**：
- [[wiki/sources/2026-04-24 UMI (Chi 2024)]]（原始论文，相对轨迹定义）
- [[wiki/sources/2026-04-24 FastUMI (Zhaxizhuoma 2025)]]（FastUMI 变换公式 Eq.1–6）
- [[wiki/entities/UMI]]（系统总览）

---

## 1. 坐标系定义

推导涉及四个坐标系：

| 符号      | 名称                 | 描述                                              | 生命周期    |
| ------- | ------------------ | ----------------------------------------------- | ------- |
| $\{W\}$ | SLAM 世界系           | 以 $t_0$ 时刻相机位姿为原点（${}^W T_C(t_0) = I$），轨迹经归一化处理 | 仅采集阶段存在 |
| $\{C\}$ | 相机系                | 中心鱼眼相机的光学中心，随手持设备运动                             | 采集 + 推理 |
| $\{F\}$ | 法兰系（`panda_link8`） | Franka 第 7 轴输出端，夹爪安装面；策略控制目标                    | 采集 + 推理 |
| $\{B\}$ | 机器人基坐标系            | 机械臂底座固定系（`panda_link0`），FK 输出的参考系               | 仅推理阶段   |


**已知常量（手眼标定）**：
- ${}^C T_F$：相机→法兰固定外参，采集时用于从 SLAM 轨迹计算法兰轨迹

**符号约定**：${}^A T_B$ 表示"坐标系 $B$ 在坐标系 $A$ 中的位姿"，即将 $B$ 中的点变换到 $A$ 中的齐次变换矩阵 $\in SE(3)$。

> ⚠️ **Matrix Studio 坐标轴**（采集可视化，与法兰系不同）：X（红）= 沿 camera0 前进方向；Y（绿）= 左；Z（蓝）= 上。这是相机系的可视化，EE-at-$t_0$ 参考下动作轴对应 `panda_link8` 定义（§9.2），调试时注意区分。

---

## 2. 采集阶段：SLAM 输出与法兰轨迹

SLAM 跟踪相机运动，轨迹以 $t_0$ 帧为参考（每段演示数据归一化使 ${}^W T_C(t_0) = I$，即 $\{W\} \equiv \{C(t_0)\}$），输出每帧相机位姿：

> [!warning] 双臂例外
> 上述“每段轨迹以各自 $t_0$ 归一化”的写法只适合单臂。双臂 UMI 不能分别把两路相机的初始位姿都设成单位阵，否则会丢失两只 gripper 的初始空间关系。双臂原始 UMI 必须先把两路 GoPro 轨迹 relocalize 到同一个 SLAM map，并转换成同一个场景锚点坐标系，再计算跨 gripper 相对位姿；完整链路见 §10。

$${}^W T_C(t) \in SE(3)$$

法兰位姿通过固定外参推算：

$${}^W T_F(t) = {}^W T_C(t) \cdot {}^C T_F$$

---

## 3. ee6d 参考定义：EE（法兰）-at-$t_0$

### 3.1 动作定义

$$\boxed{\Delta_k = {}^{F(t_0)} T_{F(t_k)} = \bigl({}^W T_F(t_0)\bigr)^{-1} \cdot {}^W T_F(t_k) \in SE(3)}$$

含义：**时刻 $t_k$ 的法兰位姿，以 $t_0$ 时刻法兰坐标系为参考**。

	初始条件：$\Delta_{t_0} = I$（单位矩阵）。

### 3.2 从 SLAM 输出直接计算（采集管道）

展开 ${}^W T_F(t) = {}^W T_C(t) \cdot {}^C T_F$：

$$\Delta_k = \bigl({}^W T_C(t_0) \cdot {}^C T_F\bigr)^{-1} \cdot {}^W T_C(t_k) \cdot {}^C T_F$$

$$= ({}^C T_F)^{-1} \cdot \underbrace{\bigl({}^W T_C(t_0)\bigr)^{-1} \cdot {}^W T_C(t_k)}_{\Delta C_k \text{（相对相机运动）}} \cdot {}^C T_F$$

---

## 4. 严谨数学推导

### 4.1 符号表

| 符号                      | 类型             | 含义                          |
| ----------------------- | -------------- | --------------------------- |
| $t_0$                   | 时刻             | 当前观测时刻（策略推理的起点）             |
| $t_k = t_0 + k\Delta t$ | 时刻             | 第 $k$ 步预测时刻                 |
| ${}^W T_C(t)$           | $SE(3)$        | SLAM 输出的相机位姿                |
| ${}^W T_F(t)$           | $SE(3)$        | 采集时法兰位姿                     |
| ${}^B T_F(t)$           | $SE(3)$        | 推理时法兰位姿（机器人 FK 输出）          |
| ${}^C T_F$              | $SE(3)$        | 相机→法兰固定外参（手眼标定值）            |
| $\Delta_k$              | $SE(3)$        | 策略预测的第 $k$ 步 EE-at-$t_0$ 动作 |
| $H$                     | $\mathbb{Z}^+$ | 预测视野（action horizon）        |

### 4.2 推导目标

给定策略预测序列 $\{\Delta_k\}_{k=1}^H$（EE-at-$t_0$ 参考），求机器人基坐标系下的目标法兰位姿序列 $\{{}^B T_F(t_k)\}_{k=1}^H$。

### 4.3 推导步骤

**Step 1**：法兰在世界系中的轨迹

$${}^W T_F(t) = {}^W T_C(t) \cdot {}^C T_F$$

**Step 2**：EE-at-$t_0$ 相对动作展开

$$\Delta_k = \bigl({}^W T_F(t_0)\bigr)^{-1} \cdot {}^W T_F(t_k)$$
	
$$= \bigl({}^W T_C(t_0) \cdot {}^C T_F\bigr)^{-1} \cdot {}^W T_C(t_k) \cdot {}^C T_F$$

$$= ({}^C T_F)^{-1} \cdot \Delta C_k \cdot {}^C T_F \tag{1}$$

（依据 $SE(3)$ 中 $(AB)^{-1} = B^{-1} A^{-1}$）

**Step 3**：相对表示的帧无关性

$\Delta_k$ 描述**物理相对运动**，与绝对坐标系（SLAM 世界系还是机器人基系）无关：

$$\Delta_k^\text{infer} = \Delta_k^\text{collect} \tag{2}$$

这是相对动作表示实现零样本跨平台迁移的根本原因：**绝对世界系消失了**。

**Step 4**：还原推理时的目标法兰位姿

推理时法兰的初始位姿为 ${}^B T_F(t_0)$（由 FK 读取），目标位姿为：

$$\boxed{{}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k} \tag{3}$$

### 4.4 分量展开（平移 + 旋转）

设 ${}^B T_F(t_0) = (\mathbf{p}_{BF_0},\ \mathbf{R}_{BF_0})$，$\Delta_k = (\mathbf{p}_{\Delta k},\ \mathbf{R}_{\Delta k})$，则：

$$\mathbf{p}_F^{(k)} = \mathbf{p}_{BF_0} + \mathbf{R}_{BF_0} \cdot \mathbf{p}_{\Delta k}$$

$$\mathbf{R}_F^{(k)} = \mathbf{R}_{BF_0} \cdot \mathbf{R}_{\Delta k}$$

---

## 5. 推理时完整变换流程

```
观测时刻 t_0
│
├─ [传感器] 获取腕部相机图像 I(t_0)
│
├─ [机器人 FK] 读取当前法兰位姿
│   ᴮT_F(t₀) ← robot.get_ee_pose()     ← 单臂只需这一项；双臂还需左右基座外参
│
├─ [策略推理] policy(I(t_0), ...) → {Δ₁, Δ₂, ..., Δ_H}
│   其中 Δ_k = ᶠ⁽ᵗ⁰⁾T_F(t_k)，初始值 Δ₀ = I
│
├─ [坐标变换] 对每个预测步 k = 1..H：
│   ᴮT_F(t_k) = ᴮT_F(t₀) · Δ_k        ← 单臂动作解码无需再次使用 ᶜT_F
│
├─ [延迟匹配] 根据 UMI PD1，选取对应执行时刻的 Δ_{k*}
│   k* = ⌈Δt_total / Δt⌉
│
└─ [逆运动学 IK] θ(t_k) = IK(ᴮT_F(t_k))
    └─ 发送关节指令给机器人控制器
```

---

## 6. 关键约束与潜在陷阱

### 6.1 ${}^C T_F$ 精度影响数据质量

EE-at-$t_0$ 方案在**数据预处理**时使用 ${}^C T_F$，而非推理时。若标定不准，误差被固化进训练数据，不会在推理时动态放大。需在数据质量检验阶段而非推理管道中处理标定误差。

### 6.2 重力方向一致性

SLAM 世界系 $\{W\}$ 通常以初始化时的重力方向定义 Z 轴（IMU 辅助）。机器人基坐标系 $\{B\}$ 同样以重力为 Z 轴。

相对表示消除了**平移原点**的差异，但若两个系的**姿态基准**（Roll/Pitch 零点）不一致，$\mathbf{R}_{\Delta k}$ 的旋转方向将出现系统性偏差。

✅ 对策：确保 SLAM 初始化时设备水平放置，与机器人 URDF 的重力定义对齐。

### 6.3 $t_0$ 选取：观测时刻 ≠ 执行时刻

UMI 的延迟匹配（PD1）指出：
- **$t_0$** = 图像被采集的时刻（观测时刻）
- **$t_\text{act}$** = 动作实际被执行的时刻（含推理延迟 + 执行延迟）

${}^B T_F(t_0)$ 必须以**图像采集时刻**的 FK 快照为准，而非发送命令时的实时状态。否则参考帧时间错位，导致轨迹偏移。

$$\Delta t_\text{total} = \underbrace{(t_\text{input} - t_\text{obs})}_\text{观测延迟} + \underbrace{(t_\text{output} - t_\text{input})}_\text{推理延迟} + \underbrace{(t_\text{act} - t_\text{output})}_\text{执行延迟}$$

发送给机器人的是 $\Delta_{k^*}$，其中 $k^* = \lceil \Delta t_\text{total} / \Delta t \rceil$。

### 6.4 SLAM 漂移的影响

相对表示 $\Delta_k$ 仅在 $t_0$ 到 $t_k$ 的**短时窗**内有效，长时 SLAM 漂移会传播进 $\Delta_k$。

✅ 对策：限制 action horizon $H$ 在合理范围（UMI 通常 $H \leq 16$ 帧），并在 FastUMI 中使用回环闭合抑制漂移。

---

## 7. 与 FastUMI 公式的对齐

FastUMI（Eq.5–6）定义的是**逐帧增量**（速度形式），与 UMI 原版的 **$t_0$ 为固定原点**的绝对相对位姿不同：

|      | UMI（Chi 2024, EE-at-$t_0$）                      | FastUMI Eq.5–6                                                                                               |
| ---- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 动作形式 | $\Delta_k = {}^{F(t_0)} T_{F(t_k)}$（$t_0$ 固定原点） | $\mathbf{p}_\text{rel}^{(i)} = \mathbf{p}_F^{(i+1)} - \mathbf{p}_F^{(i)}$（逐帧差分）                              |
| 推理公式 | ${}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k$  | ${}^B \mathbf{p}_F^{(i+1)} = {}^B \mathbf{p}_F^{(i)} + \mathbf{R}_F^{(i)} \cdot \mathbf{p}_\text{rel}^{(i)}$ |
| 漂移积累 | 不积累（每次以新 $t_0$ 为基准）                             | 积累（逐帧累加误差）                                                                                                   |
| 适配策略 | Diffusion Policy（多步预测）                          | ACT / Smooth-ACT（逐帧自回归）                                                                                      |

---

## 8. 小结

| 步骤 | 操作 | 公式 |
| --- | --- | --- |
| ① 读 FK | 获取当前法兰位姿 | ${}^B T_F(t_0)$ |
| ② 策略推理 | 输入图像，输出 ee6d | $\{\Delta_k\}_{k=1}^H$ |
| ③ 坐标变换 | EE-at-$t_0$ 参考→基坐标系 | ${}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k$ |
| ④ 延迟补偿 | 选取对应执行时刻的动作 | $k^* = \lceil \Delta t_\text{total} / \Delta t \rceil$ |
| ⑤ IK 求解 | 发送关节命令 | $\boldsymbol{\theta}(t_k) = \text{IK}({}^B T_F(t_k))$ |

**核心等式**：

$$\boxed{{}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k}$$

其中 $\Delta_k = {}^{F(t_0)} T_{F(t_k)} \in SE(3)$ 是策略预测的 ee6d，${}^B T_F(t_0)$ 由 Polymetis FK 实时读取。

---

## 9. Franka 具体实现

> **场景**：EE 定义为 Franka 法兰（`panda_link8`），Polymetis 驱动控制。

### 9.1 Franka 内部坐标系链

$${}^O T_{EE} = \underbrace{{}^O T_F}_{\substack{\text{纯运动学 FK} \\ \text{（7 个关节角决定）}}} \cdot \underbrace{{}^F T_{NE}}_{\substack{\text{法兰→名义 EE} \\ \text{（出厂固定，默认 }I\text{）}}} \cdot \underbrace{{}^{NE} T_{EE}}_{\substack{\text{用户自定义偏移} \\ \text{（可通过 Desk 设置）}}}$$

| 帧符号 | ROS TF 名称 | 含义 |
| --- | --- | --- |
| $\{O\}$ = $\{B\}$ | `panda_link0` | 机器人基坐标系 |
| $\{F\}$ | `panda_link8` | **法兰**（第 7 轴输出端，夹爪安装面） |
| $\{NE\}$ | — | 名义末端执行器（出厂写死） |
| $\{EE\}$ | `panda_EE` | 用户定义的实际 EE |

**关键结论**：当 ${}^{F}T_{NE} = I$ 且 ${}^{NE}T_{EE} = I$（Franka 出厂默认），${}^{O}T_{EE}$ 直接等于 ${}^{B}T_{F}$。

### 9.2 法兰帧的轴方向

`panda_link8` 的右手坐标系：

- **+Z**：垂直法兰面向外（安装工具的接近方向）
- **+X**：水平，指向 `panda_link7` 连接侧
- **+Y**：右手系补全（$= \hat{Z} \times \hat{X}$）

> GoPro 光轴通常与法兰 +Z 轴存在固定旋转偏移，需在手眼标定 ${}^C T_F$ 中体现。

### 9.3 读取 ${}^B T_F$（polymetis）

```python
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R
import numpy as np

robot = RobotInterface(ip_address="172.16.0.1")

def get_T_BF() -> np.ndarray:
    """读取当前法兰位姿 ᴮT_F ∈ SE(3)。"""
    pos, quat = robot.get_ee_pose()  # quat: [x, y, z, w]（scipy 约定）
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat.numpy()).as_matrix()
    T[:3, 3]  = pos.numpy()
    return T
```

**验证**：`T[:3, :3]` 应为正交旋转矩阵（行列式 = 1），`T[:3, 3]` 应为合理的平移量（单位米）。

### 9.4 数据预处理：从 SLAM 输出生成 EE-at-$t_0$ 动作

```python
import numpy as np

T_CF = np.load("camera_to_flange_extrinsic.npy")  # (4,4), ᶜT_F

# 从 SLAM 相机轨迹直接输出 EE-at-t0 格式
W_T_F_t0 = W_T_C_list[0] @ T_CF            # t0 时刻法兰位姿
W_T_F_t0_inv = np.linalg.inv(W_T_F_t0)

deltas = []
for W_T_C_tk in W_T_C_list:
    W_T_F_tk = W_T_C_tk @ T_CF
    Delta_k = W_T_F_t0_inv @ W_T_F_tk      # EE-at-t0 相对动作，初始值为 I
    deltas.append(Delta_k)
```

### 9.5 ⚠️ 标定对齐：${}^C T_F$ 的"手"必须取 `panda_link8`

手眼标定时，"手"参考点必须与 ee6d 的 $\{F\}$ 定义完全一致：

```
采集时手持 UMI                    推理时机械臂（F = panda_link8）
┌──────────────────────────┐     ┌──────────────────────────┐
│ 相机固定在 3D 打印夹爪     │     │ 相机固定在法兰            │
│ 标定：相机 → 夹爪 TCP     │  ≠  │ 标定：相机 → 法兰         │
│ 结果：ᶜT_TCP              │     │ 结果：ᶜT_F               │
└──────────────────────────┘     └──────────────────────────┘
          若两者不一致，Δ_k 的坐标意义出现系统性偏差
```

### 9.6 完整推理骨架

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from polymetis import RobotInterface

robot = RobotInterface(ip_address="172.16.0.1")

def get_T_BF() -> np.ndarray:
    pos, quat = robot.get_ee_pose()
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat.numpy()).as_matrix()
    T[:3, 3]  = pos.numpy()
    return T

def inference_step(policy, camera) -> None:
    # ① 读取 t₀ 时刻的 FK 快照（与图像时间戳对齐）
    T_BF_t0 = get_T_BF()

    # ② 策略推理 → EE-at-t₀ 格式的动作序列（初始值为 I）
    img = camera.capture()
    deltas = policy.predict(img)            # List[(4,4)]，Δ_k

    # ③ 坐标变换（无需外参）
    T_BF_targets = [T_BF_t0 @ delta_k for delta_k in deltas]

    # ④ 延迟补偿：选取对应执行时刻的动作
    k_star = compute_delay_offset()
    T_BF_exec = T_BF_targets[k_star]

    # ⑤ IK 求解 → 发送关节命令
    q_target = ik_solver.solve(T_BF_exec)
    robot.send_joint_command(q_target)
```

---

## 10. 原始 GoPro VIO UMI 的双臂坐标系统一

> **核心结论**：相对位姿只消除了共同的世界/地图坐标系，不会消除 Camera→TCP 外参、左右机器人基座外参，也不会消除 TCP 坐标轴定义差异。

本节使用 ${}^A T_B$ 表示“坐标系 $B$ 在坐标系 $A$ 中的位姿”，即把 $B$ 系坐标变换到 $A$ 系。

**来源**：
- [[wiki/sources/2026-04-24 UMI (Chi 2024)]]（map-then-localize、inter-gripper proprioception、相对轨迹）
- [UMI `06_generate_dataset_plan.py`](https://github.com/real-stanford/universal_manipulation_interface/blob/main/scripts_slam_pipeline/06_generate_dataset_plan.py)（`tx_tag_cam`、`tx_cam_tcp`、`tx_tag_tcp`）
- [UMI `real_inference_util.py`](https://github.com/real-stanford/universal_manipulation_interface/blob/main/umi/real_world/real_inference_util.py)（`tx_robot1_robot0` 与推理时跨基座变换）

### 10.1 局部符号定义

为避免与前文 Franka 法兰系 $\{F\}$ 混淆，本节将用于策略学习和动作解码的标准工具坐标系记为 $\{P_i\}$。

| 符号                | 含义                 | 获得方式             |
| ----------------- | ------------------ | ---------------- |
| $\{M\}$           | 所有双臂演示共享的 SLAM map | mapping video 构建 |
| $\{A\}$           | 桌面 ArUco/tag 场景锚点系 | mapping 阶段标定     |
| $\{C_0\},\{C_1\}$ | 右、左 GoPro 相机系      | VIO 输出           |
| $\{P_0\},\{P_1\}$ | 右、左 policy TCP     | UMI 几何定义/外参标定    |
| $\{B_0\},\{B_1\}$ | 右、左机器人基座系          | 机器人模型与双基座标定      |
| $\{E_0\},\{E_1\}$ | 机器人控制器实际 TCP       | FK 输出            |

原始仓库约定 `robot0/camera0` 为右侧，`robot1/camera1` 为左侧。自定义系统必须固定这一索引约定，不能在不同 episode 或推理时交换。

### 10.2 采集端：两路 VIO 必须定位到同一个 map

双臂采集时，两路 GoPro VIO 应得到：

$$
{}^M T_{C_0}(t),\qquad {}^M T_{C_1}(t)
$$

这里的两个 $M$ 必须是同一个地图坐标系。正确流程是：

1. 在当前场景录制一次 mapping video 并构建地图 $M$；
2. mapping 时观测场景 tag，求出 ${}^M T_A$；
3. 两只 gripper 的 demonstration video 分别 relocalize 到该地图，而不是各自新建 VIO 原点；
4. 通过统一时间戳对齐两路视频，再在同一时刻计算双手几何关系。

定义：

$$
{}^A T_M = \left({}^M T_A\right)^{-1}
$$

则两路相机在场景锚点系下为：

$$
{}^A T_{C_i}(t) = {}^A T_M\,{}^M T_{C_i}(t)
$$
原始代码对应：

```python
T_A_M = np.linalg.inv(T_M_A)       # tx_tag_slam
T_A_Ci = T_A_M @ T_M_Ci            # tx_tag_cam
```

tag 不要求在每个演示帧中可见；它主要在 mapping 阶段定义稳定的场景锚点。真正使两个 gripper 可比较的是“所有视频都重定位到同一个 map”。

> [!danger] 两个独立 VIO 原点无法直接合并
> 如果分别启动两套没有共享地图的 VIO，得到 ${}^{M_0}T_{C_0}$ 和 ${}^{M_1}T_{C_1}$，其中存在未知的 ${}^{M_0}T_{M_1}$。仅凭两条独立轨迹无法恢复该变换，必须额外引入共同 tag、地图合并/回环、运动捕捉或已知机械夹具约束。

### 10.3 每个 UMI gripper 都需要完整的 Camera→Policy-TCP 外参

为左右两个 UMI 分别定义：

$$
{}^{C_0}T_{P_0},\qquad {}^{C_1}T_{P_1}
$$

则公共场景坐标系中的 policy TCP 轨迹为：

$$
\boxed{{}^A T_{P_i}(t)
= {}^A T_M\,{}^M T_{C_i}(t)\,{}^{C_i}T_{P_i}}
$$

原始标准 UMI 根据 CAD/安装尺寸使用固定 `tx_cam_tcp`，代码中近似为纯平移：

```python
T_Ci_Pi = pose_to_mat([
    0,
    cam_to_center_height,
    cam_to_tip_offset,
    0, 0, 0,
])
T_A_Pi = T_A_Ci @ T_Ci_Pi          # tx_tag_tcp
```

这只适用于完全复现官方 GoPro 型号、安装方向和坐标轴约定的情况。自定义硬件应为两只 gripper 分别标定完整 $SE(3)$ 外参，而不是只估计平移，也不应默认两套外参完全相同。

手持 UMI 没有机器人 FK，因此这里不一定采用传统 $AX=XB$ 手眼标定。可使用已知 Target→TCP 几何关系的标定治具：GoPro 通过 PnP 求 Camera→Target，再与治具的 Target→TCP 组合，并用多姿态数据做 $SE(3)$ 最小二乘优化。

> [!warning] 检查 VIO 输出帧定义
> `camera_trajectory.csv` 的相机姿态不一定等价于 ROS optical frame；VIO/ORB-SLAM 导出脚本可能已经做过轴变换。应使用轴可视化和已知刚体运动验证 $+X,+Y,+Z$，不能仅凭变量名拼接外参。

### 10.4 训练端：inter-gripper pose 的完整变换

统一到 $A$ 系以后，右手看左手的相对位姿为：

$$
\boxed{{}^{P_0}T_{P_1}(t)
= \left({}^A T_{P_0}(t)\right)^{-1}
{}^A T_{P_1}(t)}
$$

展开为：

$$
{}^{P_0}T_{P_1}
= \left({}^{C_0}T_{P_0}\right)^{-1}
\left({}^M T_{C_0}\right)^{-1}
{}^M T_{C_1}
{}^{C_1}T_{P_1}
$$

可以看出，共同的 $A$ 系和 $M$ 系被消掉，但左右两套 Camera→TCP 外参不会被消掉。

同理，左手看右手为：

$$
{}^{P_1}T_{P_0} = \left({}^{P_0}T_{P_1}\right)^{-1}
$$

每只手自己的 EE-at-$t_0$ 动作仍为：

$$
\Delta_i(t_k)
= \left({}^A T_{P_i}(t_0)\right)^{-1}
{}^A T_{P_i}(t_k)
$$

注意，动作是“同一只手跨时间”的相对变换，inter-gripper pose 是“同一时刻跨两只手”的相对变换，两者不能混用。

### 10.5 推理端：使用双机器人基座外参重建相同观测

机器人 FK 分别给出：

$$
{}^{B_0}T_{E_0}(t),\qquad
{}^{B_1}T_{E_1}(t)
$$

若控制器 TCP $E_i$ 与训练使用的 policy TCP $P_i$ 不同，需要固定工具变换：

$$
{}^{B_i}T_{P_i}(t)
= {}^{B_i}T_{E_i}(t)\,{}^{E_i}T_{P_i}
$$

再标定双臂基座外参 ${}^{B_0}T_{B_1}$。推理时右手看左手的位姿为：

$$
\boxed{{}^{P_0}T_{P_1}(t)
= \left({}^{B_0}T_{P_0}(t)\right)^{-1}
{}^{B_0}T_{B_1}
{}^{B_1}T_{P_1}(t)}
$$

这与采集端计算的特征具有同一坐标意义，因此可以直接作为 `robot0_eef_*_wrt1` 的推理输入。

原始 UMI 的 `tx_left_right` 按“robot0=右、robot1=左”的约定表示：

$$
{}^{B_\text{left}}T_{B_\text{right}}
$$

推理代码在计算右臂坐标系中的左臂时会对它取逆。实现时建议将变量明确命名为 `T_Bleft_Bright`，避免 `left_right` 或 `robot1_robot0` 导致方向误用。

### 10.6 为什么不需要 Demo-map→Robot-base 标定

双臂策略若只使用：

- 每只手的 EE-at-$t_0$ 相对轨迹；
- 两只 policy TCP 间的相对位姿；
- 腕部 RGB 和夹爪宽度；

则不需要求：

$$
{}^{B_\text{robot}}T_{A_\text{demo}}
$$

因为采集端和推理端都在各自内部先构造相同物理含义的相对量，公共世界系会在矩阵乘法中消掉。

但若策略输入或输出包含场景锚点系中的绝对末端位姿、物体绝对位姿或固定世界方向，那么仍然需要做场景注册或 Demo→Robot 世界坐标对齐。

### 10.7 机器人腕部 Camera→TCP 标定的真正作用

固定基座机械臂推理时，inter-gripper pose 通常直接由 FK 和双基座外参计算，不需要运行腕部相机 VIO。但腕部 Camera→TCP 关系仍然决定视觉—动作几何是否与训练一致。

应尽量满足：

$$
{}^{P_i}T_{C_i}^{\text{robot}}
\approx
{}^{P_i}T_{C_i}^{\text{UMI}}
$$

手眼标定只能告诉系统数值外参，不能自动把一个明显不同视点的 RGB 图像变成训练时视点。因此优先级为：

1. 用机械安装复现 UMI Camera→Policy-TCP 几何关系；
2. 分别标定两只机器人腕部相机外参；
3. 小视角差异可配合图像增强；
4. 大视角差异需要机器人微调数据或重新训练视觉编码器。

### 10.8 Policy TCP 与控制器 TCP 不一致时的动作解码

相对动作并不对工具坐标系变化保持不变。令：

$$
X_i = {}^{E_i}T_{P_i}
$$

若策略输出 $P_i$ 系下的相对动作 $\Delta T_{P_i}$，控制器要求 $E_i$ 系下的相对动作，则：

$$
\boxed{\Delta T_{E_i}
= X_i\,\Delta T_{P_i}\,X_i^{-1}}
$$

更不容易写错的实现方式是先重建 policy TCP 的绝对目标，再转换回控制器 TCP：

$$
{}^{B_i}T_{P_i}^{des}
= {}^{B_i}T_{P_i}^{cur}\,\Delta T_{P_i}
$$

$$
{}^{B_i}T_{E_i}^{des}
= {}^{B_i}T_{P_i}^{des}\,X_i^{-1}
$$

```python
T_Bi_Pi_cur = T_Bi_Ei_cur @ T_Ei_Pi
T_Bi_Pi_des = T_Bi_Pi_cur @ Delta_Pi
T_Bi_Ei_des = T_Bi_Pi_des @ np.linalg.inv(T_Ei_Pi)
```

因此，不能通过“给相对位置加一个 TCP 偏移量”完成工具帧转换，必须使用完整的 $SE(3)$ 共轭或绝对位姿链。

### 10.9 如何标定左右机器人基座外参

推荐让两只腕部相机观测同一个固定标定板 $Q$。对第 $i$ 只机械臂：

$$
{}^{B_i}T_Q
= {}^{B_i}T_{E_i}
{}^{E_i}T_{C_i}
{}^{C_i}T_Q
$$

于是：

$$
\boxed{{}^{B_0}T_{B_1}
= {}^{B_0}T_Q
\left({}^{B_1}T_Q\right)^{-1}}
$$

应采集多组覆盖不同位置和姿态的观测，联合优化一个固定的 ${}^{B_0}T_{B_1}$，而不是只测量两个底座之间的平移距离。

### 10.10 必做一致性检查

| 检查项    | 判据/目的                                         |
| ------ | --------------------------------------------- |
| VIO 共图 | 两路轨迹使用同一 map id 和同一个 `tx_slam_tag`            |
| 时间同步   | inter-gripper pose 使用同一物理时刻的数据                |
| 外参方向   | ${}^{C_i}T_{P_i}\,{}^{P_i}T_{C_i}\approx I$   |
| 跨手互逆   | ${}^{P_0}T_{P_1}\,{}^{P_1}T_{P_0}\approx I$   |
| 基座互逆   | ${}^{B_0}T_{B_1}\,{}^{B_1}T_{B_0}\approx I$   |
| 已知夹具验证 | 两个 TCP 固定在已知距离/姿态的治具中，采集端与机器人端结果应一致           |
| 坐标轴可视化 | 分别沿 policy TCP 的 $+X,+Y,+Z$ 移动，确认训练与机器人端符号相同  |
| 旋转合法性  | 所有旋转矩阵满足 $R^TR\approx I$ 且 $\det(R)\approx 1$ |

> [!note] 左右镜像不是坐标变换
> 即使左右硬件采用镜像结构，policy TCP 仍必须使用右手坐标系。不要使用 $\det(R)=-1$ 的反射矩阵充当外参；应通过合法旋转和统一 TCP 定义处理左右轴方向。

### 10.11 双臂变换链速查

采集端：

$$
\boxed{
{}^A T_{P_i}
= {}^A T_M
{}^M T_{C_i}
{}^{C_i}T_{P_i}}
$$

$$
\boxed{
{}^{P_0}T_{P_1}
= \left({}^A T_{P_0}\right)^{-1}
{}^A T_{P_1}}
$$

推理端：

$$
\boxed{
{}^{P_0}T_{P_1}
= \left({}^{B_0}T_{E_0}{}^{E_0}T_{P_0}\right)^{-1}
{}^{B_0}T_{B_1}
{}^{B_1}T_{E_1}
{}^{E_1}T_{P_1}}
$$

最终需要标定或确定的固定量包括：

- 采集端：${}^{C_0}T_{P_0}$、${}^{C_1}T_{P_1}$；
- 推理端：${}^{B_0}T_{B_1}$、${}^{E_0}T_{P_0}$、${}^{E_1}T_{P_1}$；
- 若使用机器人腕部相机：${}^{E_0}T_{C_0}^{robot}$、${}^{E_1}T_{C_1}^{robot}$。

一句话概括：

> 采集时依靠“共同 SLAM map + 两套 Camera→UMI-policy-TCP 外参”；推理时依靠“左右机器人基座外参 + 两套 Controller-TCP→Policy-TCP 外参”。相对位姿只使采集 map 与机器人 base 不必直接对齐，并没有免除这些内部外参标定。
