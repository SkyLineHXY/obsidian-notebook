---
type: analysis
tags: [Flow Matching, 生成模型, ODE, 数学推导, Rectified Flow, CFM]
sources: [raw/sources/papers/Generative Model/Lipman 等 - 2023 - Flow Matching for Generative Modeling.pdf]
created: 2026-04-21
updated: 2026-04-21
---

# Flow Matching 完整数学推导

> **定位**：本页面是 [[wiki/concepts/Flow Matching]] 的数学展开，适合零基础读者。目标是让你不仅会用公式，还能理解每一行推导从哪里来、为什么这样做。

---

## 0. 符号定义

在阅读推导前，先把所有符号的含义固定下来：

| 符号                                                    | 含义                                                      |
| ----------------------------------------------------- | ------------------------------------------------------- |
| $d$                                                   | 数据的维度（比如动作向量的长度）                                        |
| $x \in \mathbb{R}^d$                                  | 数据空间中的一个点                                               |
| $t \in [0, 1]$                                        | 时间变量，从 0 到 1                                            |
| $p_0(x)$                                              | **源分布**，即简单的初始分布，通常取标准正态 $\mathcal{N}(0, I_d)$          |
| $q(x)$                                                | **目标分布**，即真实数据分布（我们想要学习的）                               |
| $p_t(x)$                                              | **概率路径**，$t$ 时刻的概率分布；$p_0 = \mathcal{N}(0,I)$，$p_1 = q$ |
| $\phi_t : \mathbb{R}^d \to \mathbb{R}^d$              | **流映射（Flow Map）**，把 $t=0$ 时的点推送到 $t$ 时刻                 |
| $v_t(x) : [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ | **速度场（Velocity Field）**，指定每个时空点 $(t,x)$ 的移动方向           |
| $v_\theta(t,x)$                                       | 用神经网络参数化的速度场（待学习）                                       |
| $u_t(x)$                                              | **边缘速度场**，由概率路径唯一确定的"真实"速度场                             |
| $u_t(x \mid x_1)$                                     | **条件速度场**，给定数据点 $x_1$ 时的速度场                             |
| $p_t(x \mid x_1)$                                     | **条件概率路径**，给定数据点 $x_1$ 时 $t$ 时刻的条件分布                    |
| $q(x_{1}\mid x)$                                      | 给定中间流路径数据 $x$，目标是$x_{1}$的后验概率                           |
| $x_0 \sim \mathcal{N}(0,I)$                           | 从源分布采样的噪声点                                              |
| $x_1 \sim q$                                          | 从目标数据分布采样的真实数据点                                         |
| $x_t$                                                 | $t$ 时刻的插值点，$x_t = (1-t)x_0 + t x_1$（线性插值）               |

---

## 1. 问题设定：我们想做什么？

**目标**：给定一批真实数据样本（比如机器人动作序列），学一个**生成模型**，能从中采样出新的、逼真的样本。

**直觉**：把"噪声→数据"想象成一滴墨水在水中扩散的逆过程——我们想学一个"逆时针转动"的速度场，把随机噪声"流"成真实数据的形状。

**数学框架**：用一条从 $p_0$（高斯噪声）到 $q$（真实数据）的**概率路径** $p_t$，并学一个速度场 $v_\theta(t, x)$ 来驱动这条路径。

---

## 2. 流（Flow）与常微分方程（ODE）

### 2.1 流映射的定义

**流映射** $\phi_t$ 由一个**常微分方程（ODE）**定义：

$$
\frac{d}{dt}\phi_t(x) = v_t\!\left(\phi_t(x)\right), \qquad \phi_0(x) = x
$$

**物理直觉**：想象 $x$ 是一个粒子在 $t=0$ 时的位置，$v_t(x)$ 是该点在时刻 $t$ 的速度。这个 ODE 描述粒子随时间的运动轨迹。
- $\phi_0(x) = x$：$t=0$ 时粒子在原地
- $\phi_1(x)$：$t=1$ 时粒子到达的最终位置

如果 $x_0 \sim p_0$（从高斯分布采样），那么 $\phi_1(x_0) \sim p_1$，我们希望 $p_1 = q$。

### 2.2 概率分布如何随流演变

设 $X_0 \sim p_0$，令 $X_t = \phi_t(X_0)$，则 $X_t$ 的概率密度 $p_t$ 满足**连续性方程（Continuity Equation）**：

$$
\frac{\partial p_t(x)}{\partial t} + \nabla_x \cdot \!\left(p_t(x)\, v_t(x)\right) = 0
$$

**推导来源**：这是流体力学中的质量守恒定律。$\nabla_x \cdot (\cdot)$ 是散度算子，表示"单位体积内概率的净流出量"。概率不会凭空消失，所以时间上的减少量 = 空间上的流出量，符号取负即得上式。

**意义**：这个方程将**概率路径** $p_t$ 与**速度场** $v_t$ 唯一绑定在一起。只要给定一条从 $p_0$ 到 $q$ 的概率路径 $\{p_t\}_{t\in[0,1]}$，在某种意义下就存在一个对应的速度场 $u_t$（称为**边缘速度场**）驱动它。

---

## 3. 训练目标：Flow Matching（FM）损失

### 3.1 核心思路

我们不直接推导解析的 $u_t$，而是**让神经网络 $v_\theta$ 回归 $u_t$**。损失函数是：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x \sim p_t(x)} \left\| v_\theta(t, x) - u_t(x) \right\|_2^2
$$

**解读**：随机采一个时刻 $t$，再从该时刻的边缘分布 $p_t$ 里采一个点 $x$，要求神经网络预测的速度 $v_\theta(t,x)$ 尽量接近真实速度 $u_t(x)$。

### 3.2 为什么 FM 损失不可直接计算？

**关键困难**：$u_t(x)$ 是**边缘速度场**，它的表达式要对所有数据点做积分：
（**在t时刻、位置 $x$处的粒子，不知道自己的"目标终点"是哪个 $x_{1}$。边缘速度场$u_t(x)$ 是对所有可能的目标方向取一个加权平均。**）
$$
q(x_{1}|x)=\frac{p_t(x \mid x_1)\, q(x_1)}{p_t(x)}
$$
$$
u_t(x) = \int u_{t}(x\mid x_{1})q(x_{1}\mid x)dx_{1}=  \int u_t(x \mid x_1)\, \frac{p_t(x \mid x_1)\, q(x_1)}{p_t(x)}\, dx_1
$$

其中 $p_t(x) = \int p_t(x \mid x_1)\, q(x_1)\, dx_1$。

- 我们没有 $q(x_1)$ 的解析表达式（只有数据样本）
- 对高维空间的积分在实践中不可处理

因此，**直接优化 $\mathcal{L}_{\text{FM}}$ 是不可行的**。

---

## 4. 关键突破：条件 Flow Matching（CFM）

### 4.1 构造条件概率路径

**思路**：与其处理难以计算的边缘分布，不如**以单个数据点 $x_1$ 为条件**，构造一条从噪声到 $x_1$ 的"小路径"。

对每个数据点 $x_1$，定义**条件概率路径** $p_t(x \mid x_1)$，满足：

$$
p_0(x \mid x_1) = \mathcal{N}(x;\, 0, I), \qquad p_1(x \mid x_1) = \delta(x - x_1)
$$

**直觉**：$t=0$ 时从高斯噪声出发，$t=1$ 时"集中"到数据点 $x_1$ 处。

边缘概率路径可以写成对所有数据点的平均：
$$
p_t(x) = \int p_t(x \mid x_1)\, q(x_1)\, dx_1
$$

**验证端点**：
- $t=0$：$p_0(x) = \int \mathcal{N}(x;0,I)\, q(x_1)\, dx_1 = \mathcal{N}(x;0,I) = p_0$ ✓
- $t=1$：$p_1(x) = \int \delta(x - x_1)\, q(x_1)\, dx_1 = q(x)$ ✓

### 4.2 条件速度场

设 $u_t(x \mid x_1)$ 是驱动 $p_t(x \mid x_1)$ 的条件速度场（通过条件概率路径的连续性方程确定）。

**边缘速度场与条件速度场的关系**：

$$
\boxed{u_t(x) = \int u_t(x \mid x_1)\, \frac{p_t(x \mid x_1)\, q(x_1)}{p_t(x)}\, dx_1 = \int u_t(x \mid x_1)q(x_{1}|x)dx_{1} =\mathbb{E}_{x_1 \sim q}\!\left[u_t(x \mid x_1) \,\Big|\, x\right]}
$$

即：边缘速度场 = 条件速度场的**后验期望**。

### 4.3 CFM 损失

将"给定数据点 $x_1$，回归条件速度场"定义为 **CFM 损失**：

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\; x_1 \sim q,\; x \sim p_t(x \mid x_1)} \left\| v_\theta(t, x) - u_t(x \mid x_1) \right\|_2^2
$$

这个损失**完全可以计算**：只需从数据集中采样 $x_1$，再从条件路径采样 $x$。

---

## 5. 核心等价定理：CFM = FM（梯度相同）

### 5.1 定理陈述

$$
\nabla_\theta\, \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta\, \mathcal{L}_{\text{CFM}}(\theta)
$$

即：**可计算的 CFM 损失与不可计算的 FM 损失，对参数 $\theta$ 的梯度完全相同**。

这意味着我们可以用 CFM 损失进行训练，得到与最小化 FM 损失完全等价的结果。

### 5.2 完整证明

**第一步**：展开 FM 损失

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x \sim p_t}\!\left[\|v_\theta(t,x)\|^2 - 2\langle v_\theta(t,x),\, u_t(x) \rangle + \|u_t(x)\|^2\right]
$$

第三项 $\|u_t(x)\|^2$ 不含 $\theta$，对梯度无贡献，所以：

$$
\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathbb{E}_{t, x}\!\left[\|v_\theta(t,x)\|^2 - 2\langle v_\theta(t,x),\, u_t(x) \rangle\right] \tag{A}
$$

**第二步**：展开 CFM 损失

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_1, x}\!\left[\|v_\theta(t,x)\|^2 - 2\langle v_\theta(t,x),\, u_t(x \mid x_1) \rangle + \|u_t(x \mid x_1)\|^2\right]
$$

第三项同样不含 $\theta$，故：

$$
\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathbb{E}_{t, x_1, x}\!\left[\|v_\theta(t,x)\|^2 - 2\langle v_\theta(t,x),\, u_t(x \mid x_1) \rangle\right] \tag{B}
$$

**第三步**：比较两式的交叉项

对 (A) 中的交叉项，利用边缘分布 $p_t(x) = \int p_t(x|x_1) q(x_1) dx_1$：  
$$
\mathbb{E}_{t, x \sim p_t}\!\left[\langle v_\theta(t,x),\, u_t(x) \rangle\right]
= \mathbb{E}_{t, x}\!\left\langle v_\theta(t,x),\, \underbrace{\mathbb{E}[u_t(x|x_1)|x]}_{= u_t(x)} \right\rangle
$$

将后验期望展开：

$$
= \mathbb{E}_{t, x}\!\left\langle v_\theta(t,x),\, \int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)} dx_1 \right\rangle
$$

$$
= \int\!\!\int p_t(x)\, \left\langle v_\theta(t,x),\, \int u_t(x|x_1)\frac{p_t(x|x_1)q(x_1)}{p_t(x)} dx_1 \right\rangle dx\, dt
$$

$$
= \int\!\int\!\int \left\langle v_\theta(t,x),\, u_t(x|x_1) \right\rangle p_t(x|x_1)\, q(x_1)\, dx\, dx_1\, dt
$$

$$
= \mathbb{E}_{t, x_1 \sim q, x \sim p_t(x|x_1)}\!\left[\langle v_\theta(t,x),\, u_t(x|x_1) \rangle\right]
$$

这正好是 (B) 中的交叉项。

**第四步**：比较二次项

(A) 和 (B) 中的二次项 $\|v_\theta(t,x)\|^2$ 都关于 $x$ 取边缘期望 $\mathbb{E}_{x \sim p_t}$，这是因为：

$$
\mathbb{E}_{x_1, x \sim p_t(x|x_1)}\!\left[\|v_\theta(t,x)\|^2\right] = \mathbb{E}_{x \sim p_t(x)}\!\left[\|v_\theta(t,x)\|^2\right]
$$

（对联合分布的期望等于对边缘分布的期望。）

**结论**：(A) = (B)，即 $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$。$\blacksquare$

---

## 6. Rectified Flow：最简的条件路径选择

### 6.1 线性插值路径

**选择**：对 $(x_0, x_1)$ 对，令条件路径为**线性插值**：

$$
x_t = (1-t)\, x_0 + t\, x_1, \qquad x_0 \sim \mathcal{N}(0, I),\; x_1 \sim q
$$

这定义了一个条件分布：

$$
p_t(x \mid x_1) = \mathcal{N}\!\left(x;\; t\, x_1,\; (1-t)^2 I\right)
$$

**验证端点**：
- $t=0$：$p_0(x|x_1) = \mathcal{N}(x; 0, I)$ ✓
- $t=1$：$p_1(x|x_1) = \mathcal{N}(x; x_1, 0) = \delta(x - x_1)$ ✓

### 6.2 条件速度场的推导

对线性路径，条件速度场由路径对时间的导数直接给出：

$$
u_t(x_t \mid x_0, x_1) = \frac{d x_t}{dt} = x_1 - x_0
$$

**关键性质**：这个速度场**与时间 $t$ 无关**，是一个常数向量 $(x_1 - x_0)$！

**物理直觉**：粒子从 $x_0$ 出发，以恒定速度 $(x_1 - x_0)$ 直线飞向 $x_1$。路径是完美的直线。

### 6.3 Rectified Flow 训练目标

代入 CFM 损失，得到 Rectified Flow 的训练目标：

$$
\boxed{
\mathcal{L}_{\text{RF}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x_0 \sim \mathcal{N}(0,I),\; x_1 \sim q} \left\| v_\theta\!\left(t,\, x_t\right) - (x_1 - x_0) \right\|_2^2
}
$$

其中 $x_t = (1-t)x_0 + tx_1$。

**解读**：
- 从数据集里采样一个真实样本 $x_1$
- 从高斯分布里采样一个噪声点 $x_0$
- 随机采样一个时刻 $t$
- 在 $t$ 时刻的插值点 $x_t$ 处，让网络预测"应该往哪个方向走"
- 答案就是最简单的：往 $x_1 - x_0$ 方向走

---

## 7. 完整训练与推理算法

### 7.1 训练算法（伪代码）

重复以下步骤直到收敛：
1. 从数据集采样一批真实数据 $x_1 \sim q$
2. 采样对应噪声 $x_0 \sim \mathcal{N}(0, I)$（形状与 $x_1$ 相同）
3. 均匀采样时刻 $t \sim \mathrm{Uniform}[0, 1]$
4. 计算插值点 $x_t = (1-t) x_0 + t x_1$
5. 计算损失 $L = \lVert v_\theta(t, x_t) - (x_1 - x_0) \rVert^2$
6. 反向传播，更新参数 $\theta$

**对比 DDPM 训练的简洁性**：
- DDPM 需要 noise schedule（$\bar{\alpha}_t$、$\beta_t$ 等参数）
- DDPM 需要 SNR 加权
- Rectified Flow 只需均匀采 $t$，目标 $(x_1 - x_0)$ 恒定，**一个超参数都没有**

### 7.2 推理算法（ODE 求解）

**目标**：从 $x_0 \sim \mathcal{N}(0, I)$ 出发，用学好的 $v_\theta$ 驱动 ODE，积分到 $t=1$ 得到生成样本。

**Euler 方法（最简单）**：

将 $[0, 1]$ 均匀分成 $N$ 步，步长 $h = 1/N$：

$$
x_{t_{k+1}} = x_{t_k} + h \cdot v_\theta(t_k,\, x_{t_k}), \qquad t_k = k/N,\; k=0,1,\ldots,N-1
$$

**为什么 Rectified Flow 推理步数少？**

因为训练目标 $(x_1 - x_0)$ 是常数，理想情况下速度场 $v_\theta(t, x)$ 应该沿整条路径不变。这意味着 ODE 的轨迹**非常接近直线**，数值积分误差极小，通常 **1~10 步** 就足够。

对比 DDPM：前向加噪过程用的是随机 SDE，反向去噪轨迹是弯曲的，需要 100~1000 步。

---

## 8. 与扩散模型的统一视角

### 8.1 扩散模型是 Flow Matching 的特例

DDPM 使用的条件路径形式为：

$$
p_t(x \mid x_1) = \mathcal{N}(x;\; \alpha_t\, x_1,\; \sigma_t^2\, I)
$$

其中 $\alpha_t, \sigma_t$ 由 noise schedule 决定。

对应的条件速度场：

$$
u_t(x_t \mid x_1) = \frac{d\mu_t}{dt} + \frac{d\sigma_t}{dt} \cdot \frac{x_t - \mu_t}{\sigma_t}
= \dot{\alpha}_t\, x_1 + \dot{\sigma}_t\, \varepsilon
$$

其中 $\varepsilon = (x_t - \alpha_t x_1)/\sigma_t$ 是预测的噪声（即 DDPM 里的 $\epsilon$-prediction）。

**结论**：DDPM 的 $\epsilon$-prediction 训练目标，本质上是 Flow Matching 在特定 Gaussian 路径选择下的 CFM 损失的等价写法。Flow Matching 提供了一个**更统一的视角**，让我们在不同路径选择（线性 vs 余弦 vs DDPM 路径）之间自由切换。

### 8.2 为什么线性路径更好？

线性路径（Rectified Flow）相比扩散模型路径的优势：

| 维度 | 扩散路径 | 线性路径（Rectified Flow） |
|---|---|---|
| 路径形状 | 弯曲（依赖 noise schedule） | **直线** |
| 速度场方向 | 随 $t$ 变化 | **恒定** $(x_1 - x_0)$ |
| ODE 轨迹曲率 | 高 | **低** |
| 需要积分步数 | 100-1000 | **1-10** |
| 训练超参数 | noise schedule | **无** |

---

## 9. 在机器人策略中的意义

在机器人学习（如 [[wiki/concepts/Diffusion Policy]] → Flow Matching Policy）中：

- **$x_1$** = 目标动作序列（来自专家演示数据集）
- **$x_0$** = 随机噪声（形状与动作序列相同）
- **$v_\theta(t, x_t, o)$** = 以机器人观测 $o$（图像、关节状态）为条件的速度场网络

推理时：机器人采样一个噪声动作 $x_0$，用 1~4 步 Euler 积分生成动作序列，实时控制频率可达 **20-50 Hz**，远优于扩散模型的 5-10 Hz。

这就是为什么 Flow Matching 正在取代 DDPM 成为机器人策略的默认生成模型参数化。

---

## 相关页面

- [[wiki/concepts/Flow Matching]] — 概念概览与与扩散模型对比
- [[wiki/concepts/DDPM]] — 扩散模型，Flow Matching 的"前辈"
- [[wiki/concepts/Diffusion Policy]] — 机器人策略中的扩散模型应用
- [[RL 微调表达性策略方法对比]] — FM 策略的 RL 微调挑战与各家解法
