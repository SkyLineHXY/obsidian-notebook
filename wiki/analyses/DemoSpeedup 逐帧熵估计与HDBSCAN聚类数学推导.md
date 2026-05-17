---
type: analysis
tags: [DemoSpeedup, KDE, HDBSCAN, InformationTheory, DensityClustering, ImitationLearning, ActionEntropy]
sources: [raw/sources/papers/IL(Imitation Learning)/Guo 等 - 2025 - DemoSpeedup Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration/Guo 等 - 2025 - DemoSpeedup Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration.md]
created: 2026-04-24
updated: 2026-04-24
---

# DemoSpeedup：逐帧动作熵估计 & HDBSCAN 聚类数学推导

> 本页为 [[wiki/sources/2026-04-23 DemoSpeedup]] 核心两步骤的严谨数学展开，覆盖 Gaussian KDE 密度估计、条件熵计算，以及 HDBSCAN 从核心距离到聚类稳定性的完整推导链。

---

## 严谨数学推导

---

## 一、逐帧动作熵估计（Gaussian KDE）

### 1.1 符号定义

| 符号                                     | 类型   | 物理/算法含义                             |
| -------------------------------------- | ---- | ----------------------------------- |
| $o_t$                                  | 向量   | 时刻 $t$ 的观测（图像 + 本体感知状态）             |
| $A_t = \{a_t[t], \ldots, a_t[t+K-1]\}$ | 序列   | 动作块（Action Chunk），长度 $K$            |
| $\pi_\theta(A_t \mid o_t)$             | 条件分布 | 代理策略（ACT 或 DP 均可）                   |
| $N$                                    | 整数   | 每帧的动作块采样次数                          |
| $a_j^i[t]$                             | 标量   | 第 $i$ 个采样、时刻 $j$ 的动作块中对应第 $t$ 步的动作值 |
| $K$                                    | 整数   | Action Chunk 长度（块内动作数）              |
| $h$                                    | 正实数  | Gaussian 核带宽（控制平滑程度）                |
| $\hat{p}(a_t \mid o_t)$                | 标量   | 条件动作概率密度估计值                         |
| $\hat{H}(a_t \mid o_t)$                | 标量   | 条件动作熵估计值（最终输出，每帧一个数）                |

### 1.2 推导前提：为什么需要 KDE？

**核心障碍**：示教轨迹中，每个时刻 $t$ 只记录了**一个**人类动作 $a_t$。单个点无法估计任何分布，更无法计算熵。

**解决思路**：借助代理策略 $\pi_\theta$ 的**生成能力**——给定同一观测 $o_t$，反复采样 $N$ 次，得到 $N$ 个假设动作：

$$\left\{ a_t^1[t],\; a_t^2[t],\; \ldots,\; a_t^N[t] \right\} \sim \pi_\theta(\cdot \mid o_t)$$

这 $N$ 个样本反映了策略对"当前观测下应该采取什么动作"的**不确定程度**——这正是动作熵的语义。

**目标**：用这 $N \cdot K$ 个样本，通过 KDE 估计出连续概率密度 $\hat{p}(a_t \mid o_t)$，再计算熵 $\hat{H}(a_t \mid o_t)$。

---

### 1.3 标准 Gaussian KDE 理论

**核密度估计（Kernel Density Estimation, KDE）** 是一种非参数密度估计方法。给定 $n$ 个独立同分布样本 $\{x_1, x_2, \ldots, x_n\}$，用核函数 $K(\cdot)$ 估计真实概率密度：

$$\hat{p}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right)$$

选取 **Gaussian 核函数**：

$$K(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$$

代入后，标准 Gaussian KDE 公式为：

$$\hat{p}(x) = \frac{1}{nh} \sum_{i=1}^{n} \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{(x - x_i)^2}{2h^2}\right) \tag{KDE-0}$$

**物理直觉**：每个样本点 $x_i$ 贡献一个以自身为中心、标准差为 $h$ 的 Gaussian"小山包"，所有小山包叠加后即为连续的密度估计曲线。

**带宽 $h$ 的作用**：

| 带宽取值   | 效果             | 后果               |
| ------ | -------------- | ---------------- |
| $h$ 过大 | 过度平滑，各样本高斯重叠严重 | 密度"摊平"，估计熵偏高（虚高） |
| $h$ 过小 | 欠平滑，每个样本独立成峰   | 密度过于尖锐，估计熵偏低（虚低） |
| $h$ 适中 | 平衡拟合与平滑        | 熵估计可靠 ✓          |

经典的 **Silverman 经验法则**给出一维最优带宽：
$$h^* = 1.06 \,\hat{\sigma}\, n^{-1/5}$$
其中 $\hat{\sigma}$ 为样本标准差，$n$ 为样本数。

---
### 1.4 Step 1：应用 KDE 估计条件动作密度

在 DemoSpeedup 中，对每帧 $t$，构造样本集合：
$$\mathcal{S}_t = \left\{ a_j^i[t] \;\middle|\; j \in [t-K+1, t],\; i \in [1, N] \right\}$$

样本总数为 $|\mathcal{S}_t| = N \cdot K$。其中求和跨越两个维度：

- **$i$（采样维度）**：对同一时刻 $t$ 的观测，代理策略采样 $N$ 次，引入策略的随机性
- **$j$（块内维度）**：利用 Action Chunking 的时序重叠特性——时刻 $j$ 的动作块 $A_j$ 包含对未来第 $t$ 步的预测 $a_j^i[t]$，聚合 $K$ 个相邻时刻的预测可以增加有效样本量、降低估计方差

将 $\mathcal{S}_t$ 代入式 (KDE-0)，令 $n = NK$，得到**条件动作密度估计**：

$$\boxed{\hat{p}(a_t \mid o_t) = \frac{1}{NKh} \sum_{j=t-K+1}^{t} \sum_{i=1}^{N} \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{(a_t - a_j^i[t])^2}{2h^2}\right)} \tag{1}$$

**物理意义**：式 (1) 给出了"在当前观测 $o_t$ 下，动作值恰好等于 $a_t$ 的可能性有多高"的连续估计。

---

### 1.5 Step 2：条件动作熵计算

连续随机变量的**微分熵**（Differential Entropy）定义为：

$$H(a_t \mid o_t) = -\int \hat{p}(a \mid o_t) \log \hat{p}(a \mid o_t) \, da \tag{连续熵}$$

由于 $\hat{p}$ 没有解析积分形式，用蒙特卡洛（Monte Carlo）近似——以 $\mathcal{S}_t$ 中的样本点作为积分的离散支撑点：

$$\int f(a) \, da \approx \frac{1}{NK} \sum_{j,i} f\!\left(a_j^i[t]\right)$$

代入连续熵公式，并将归一化常数 $\frac{1}{NK}$ 吸收（相对大小比较时可忽略），得到**条件动作熵估计**：

$$\boxed{\hat{H}(a_t \mid o_t) = -\sum_{j=t-K+1}^{t} \sum_{i=1}^{N} \hat{p}\!\left(a_j^i[t] \mid o_t\right) \log \hat{p}\!\left(a_j^i[t] \mid o_t\right)} \tag{2}$$

---

### 1.6 关键性质验证

**性质 1：低熵 ↔ 策略行为一致（精密段）**

若所有 $N \cdot K$ 个样本 $a_j^i[t]$ 都集中在某个值 $a^*$ 附近：

- $\hat{p}(a^* \mid o_t)$ 极大（密度峰值集中）
- 其他位置 $\hat{p} \approx 0$
- $-p \log p$ 仅在 $p \approx p_{max}$ 时贡献，总和 $\approx -1 \cdot \log 1 = 0$（自然对数）

$$\Rightarrow \hat{H}(a_t \mid o_t) \to 0 \quad \text{（低熵）}$$

**性质 2：高熵 ↔ 策略行为多样（随意段）**

若 $N \cdot K$ 个样本均匀分散在 $[a_{min}, a_{max}]$ 上（均匀分布），KDE 趋近于均匀密度 $p = \frac{1}{a_{max}-a_{min}}$：

$$H_{\text{uniform}} = \log(a_{max} - a_{min}) \quad \text{（微分熵最大化）}$$

即：动作越随意、越"没有特定偏好"，熵越高。

---

### 1.7 ACT vs DP 的采样方式差异

| 策略                | 采样机制               | 数学表达                                                                          |
| ----------------- | ------------------ | ----------------------------------------------------------------------------- |
| **ACT**（CVAE）     | 从标准正态先验采样潜变量，解码为动作 | $z^i \sim \mathcal{N}(0, I)$，$a_t^i = \text{Decoder}(z^i, o_t)$               |
| **DP**（DDPM/DDIM） | 采样不同初始噪声，执行完整去噪链   | $\epsilon^i \sim \mathcal{N}(0, I)$，$a_t^i = \text{Denoise}(\epsilon^i, o_t)$ |

两者的共同点：给定**相同的** $o_t$，通过随机性来源（潜变量/噪声）生成 $N$ 个不同动作样本，从而探测策略的条件分布宽度。

---

## 二、HDBSCAN 密度聚类数学推导

### 2.1 符号定义

| 符号                         | 类型  | 含义                                            |
| -------------------------- | --- | --------------------------------------------- |
| $X = \{x_1, \ldots, x_n\}$ | 点集  | 待聚类数据（每帧归一化后的熵值 + 时间索引，二维）                    |
| $d(a, b)$                  | 标量  | 欧氏距离 $\|a - b\|_2$                            |
| $m_{pts}$                  | 整数  | 核心距离参数（计算 $d_{core}$ 时用的邻居数）                  |
| $m_{clSize}$               | 整数  | 最小聚类大小（压缩层次树的阈值）                              |
| $N^{(m)}(x)$               | 点   | $x$ 的第 $m$ 近邻                                 |
| $d_{core,m}(x)$            | 标量  | 点 $x$ 的核心距离（local density 的逆指标）               |
| $d_{mreach}(a,b)$          | 标量  | 点 $a, b$ 间的互达距离（Mutual Reachability Distance） |
| $T^*$                      | 树   | 基于互达距离的最小生成树（MST）                             |
| $\lambda$                  | 标量  | 密度水平，$\lambda = 1/\epsilon$（$\epsilon$ 为连接半径） |
| $\lambda_{birth}(C)$       | 标量  | 聚类 $C$ 从父聚类分裂出现时的密度水平                         |
| $\lambda_{out}(x, C)$      | 标量  | 点 $x$ 离开聚类 $C$ 时的密度水平                         |
| $S(C)$                     | 标量  | 聚类 $C$ 的稳定性分数                                 |
| $P$                        | 集合  | 精密集（低熵聚类时间索引）                                 |
| $C$                        | 集合  | 随意集（高熵离群点时间索引）                                |

### 2.2 推导前提：DBSCAN 的局限

HDBSCAN 是 DBSCAN 的层次化推广。先回顾 DBSCAN：

**DBSCAN 核心概念**（参数：$\epsilon$，$m_{pts}$）：

- $\epsilon$-邻域：$N_\epsilon(p) = \{q : d(p,q) \leq \epsilon\}$
- 核心点：$|N_\epsilon(p)| \geq m_{pts}$
- 密度可达（Density-Reachable）：$q$ 可从 $p$ 经核心点链到达
- 聚类：互相密度可达的最大点集；不可达点为噪声

**DBSCAN 的根本问题**：固定 $\epsilon$ 无法适应变密度分布。

在机器人轨迹的熵序列中：

- 精密抓取段：熵值**密集**集中在低值区域
- 过渡段：熵值**稀疏**分散在中高值区域
- 同一 $\epsilon$ 无法同时正确识别两种密度

HDBSCAN 的解法：**在所有 $\epsilon$ 尺度上运行 DBSCAN，提取最稳定的聚类结构**。

---

### 2.3 Step 1：核心距离（Core Distance）

对每个点 $x \in X$，定义其**核心距离**：

$$d_{core,m}(x) = d\!\left(x,\; N^{(m_{pts})}(x)\right) \tag{3}$$

即 $x$ 到其**第 $m_{pts}$ 近邻**的距离。

**物理意义**：$d_{core,m}(x)$ 是以 $x$ 为中心、刚好包含 $m_{pts}$ 个点所需的最小半径，反映 $x$ 所在区域的**局部密度**的倒数：

$$d_{core,m}(x) \text{ 小} \;\Leftrightarrow\; x \text{ 所在区域密度高（邻居近）}$$
$$d_{core,m}(x) \text{ 大} \;\Leftrightarrow\; x \text{ 所在区域密度低（邻居远，可能是噪声/离群点）}$$

---

### 2.4 Step 2：互达距离（Mutual Reachability Distance）

在核心距离的基础上，定义两点之间的**互达距离**：

$$d_{mreach,m}(a, b) = \max\!\left\{\; d_{core,m}(a),\quad d_{core,m}(b),\quad d(a, b) \;\right\} \tag{4}$$

**逐步推导其关键性质**：

**性质 1：对称性**

$$d_{mreach}(a,b) = \max\{d_{core}(a), d_{core}(b), d(a,b)\} = \max\{d_{core}(b), d_{core}(a), d(b,a)\} = d_{mreach}(b,a)$$

→ 满足度量对称性。

**性质 2：密集区域退化**

若 $a, b$ 都处于高密度区域，则 $d_{core}(a)$ 和 $d_{core}(b)$ 均很小，可能小于 $d(a,b)$：

$$d_{mreach}(a,b) = \max\{\underbrace{d_{core}(a)}_{\text{小}},\; \underbrace{d_{core}(b)}_{\text{小}},\; d(a,b)\} = d(a,b)$$

→ 在高密度区域，互达距离**退化为欧氏距离**，保持原有度量关系。

**性质 3：稀疏区域惩罚（关键！）**

若 $a$ 或 $b$ 处于低密度区域（如高熵的随意段），$d_{core}$ 会很大：

$$d_{mreach}(a,b) = \max\{d_{core}(a), d_{core}(b), d(a,b)\} \geq d_{core}(a) \gg d(a,b)$$

→ 即使两点在欧氏空间中很近，若其中之一处于稀疏区，互达距离也会被**"抬高"**，使连接代价增大，自然地将稀疏点隔离。

**设计直觉**：这一机制使得 MST 在稀疏区域"不愿意"构建边，最终使稀疏点成为离群点而非聚类成员。

---

### 2.5 Step 3：最小生成树（MST）

以 $X$ 中全部 $n$ 个点为节点，以 $d_{mreach}(a,b)$ 为边权，构造完全图 $G = (X, E)$。

用 Prim 或 Kruskal 算法求**最小生成树**：

$$T^* = \underset{\substack{T \subseteq G \\ T \text{ 是 } X \text{ 上的生成树}}}{\arg\min} \sum_{(a,b) \in T} d_{mreach}(a, b) \tag{5}$$

**性质**：MST 恰好有 $n-1$ 条边，且每条边都是"最经济"地连接两个已有连通分量的桥梁。

**物理意义**：MST 的每条边代表两个点（或点集）之间最难跨越的"密度屏障"——边权越大，说明连接这两部分需要穿越越稀疏的区域。

---

### 2.6 Step 4：层次树（Dendrogram）构建

将 $T^*$ 的 $n-1$ 条边按边权**从大到小**排序，依次删除：

- 删除第 $k$ 条边（权重 $w_k$）意味着：在密度水平 $\lambda_k = 1/w_k$ 处，原本连通的两个分量发生**分裂**

这个过程构建出一棵**层次树（Dendrogram）**：

$$\text{树节点} = \text{数据点或分量}, \quad \text{内部节点} = \text{分裂事件（密度水平 } \lambda_k \text{）}$$

```
λ 高（稀疏连接先被切断）
         ┌────────────┐
         │  $\lambda_1$（全局最稀疏连接断裂）
         │          │
    ┌────┴────┐  ┌──┴────┐
    │$\lambda_2$│  │$\lambda_3$│
   ┌┴┐       ···  ···  ···
  C₁ C₂   ← λ 低（密集连接，最后断裂）
```

树叶是单个数据点，树根是全集，每个内部节点记录分裂时的 $\lambda$ 值。

---

### 2.7 Step 5：聚类树压缩（Condensed Tree）

原始 Dendrogram 中存在大量"单点掉落"（一个点从大分量中脱离），这是噪声而非真实分裂。

**压缩规则**：

- 若分裂后，某子分量的点数 $< m_{clSize}$：
  - 该子分量的点被标记为"从当前聚类中**脱落**"的噪声
  - 记录其脱落时的密度水平 $\lambda_{out}(x, C_{\text{parent}})$
- 若分裂后，**两个**子分量均 $\geq m_{clSize}$：
  - 视为真正的聚类分裂，产生两个独立子聚类

压缩后只保留"真正分裂"节点，所有噪声点以脱落事件记录。

---

### 2.8 Step 6：聚类稳定性与最优选择（FOSC 算法）

对压缩树中的每个聚类 $C$，定义**稳定性分数**：

$$S(C) = \sum_{x \in C} \bigl(\lambda_{out}(x, C) - \lambda_{birth}(C)\bigr) \tag{6}$$

**各项物理意义**：

| 项 | 含义 |
|---|---|
| $\lambda_{birth}(C) = 1/\epsilon_{birth}$ | 聚类 $C$ 从父聚类**分裂出现**时的密度水平（固定值） |
| $\lambda_{out}(x, C)$ | 点 $x$ **离开**聚类 $C$ 的密度水平（各点不同：或因聚类再次分裂，或因脱落为噪声） |
| $\lambda_{out}(x,C) - \lambda_{birth}(C)$ | 点 $x$ 在聚类 $C$ 中"存活"的密度跨度（越大 = 越稳定） |

**直觉**：$S(C)$ 是各点存活时长在密度轴上的**积分面积**——聚类在宽广密度范围内稳定存在 → $S$ 大 → 真实聚类。

**最优聚类选择（自下而上贪心，FOSC）**：

从压缩树叶节点向根遍历，对每个内部节点（聚类 $C$ 与其子聚类 $\{C_1, C_2, \ldots\}$）：

$$\text{决策规则：} \begin{cases} \text{保留 } C \text{（不分裂）} & \text{若 } S(C) \geq \sum_{C' \in \text{children}(C)} S^*(C') \\ \text{选子聚类} & \text{若 } S(C) < \sum_{C' \in \text{children}(C)} S^*(C') \end{cases} \tag{7}$$

其中 $S^*(C')$ 是子聚类的最优选择后的有效稳定性。

**保证**：该贪心算法能找到全局最大化总稳定性的平面聚类划分。

---

### 2.9 Step 7：离群点识别与分数

未被任何稳定聚类选中的点被标记为**离群点/噪声**，并赋予 GLOSH 分数（Global-Local Outlier Score from the Hierarchy）：

$$\text{outlier\_score}(x) = 1 - \frac{\lambda_{out}(x,\, C_{\text{assigned}})}{\lambda_{\max}(x)} \tag{8}$$

- $\lambda_{out}(x, C_{\text{assigned}})$：点 $x$ 在其最终所属（或脱落时的）聚类中存活的密度水平
- $\lambda_{\max}(x)$：$x$ 在整个层次树中能达到的最高密度水平

**分数范围**：$[0, 1]$，越接近 $1$ 越像离群点。

---

### 2.10 在 DemoSpeedup 中的完整应用流程

**输入**：每帧的熵估计值 $\hat{H}(a_t \mid o_t)$，拼接时间索引 $t/T$，归一化为二维点：

$$x_t = \left(\frac{\hat{H}(a_t \mid o_t) - \mu}{\sigma},\; \frac{t}{T}\right)$$

时间索引确保聚类保留**时序结构**，避免将不连续的高/低熵段错误地合并。

**Isolation Forest 预处理**（HDBSCAN 之前）：

先用孤立森林（Isolation Forest）检测极端异常熵值（遥操作噪声导致的突变）并替换为相邻正常值，避免噪声污染聚类结果。

**HDBSCAN 结果解读**：

| HDBSCAN 输出 | 条件 | DemoSpeedup 标签 |
|---|---|---|
| 稳定聚类 | 聚类均值熵 $< 0$（归一化后低熵） | **精密集 $P$**：保留原速或轻微下采样 |
| 稳定聚类（被过滤） | 聚类均值熵 $\geq 0$ | 视为过渡噪声，排除 |
| 离群点 | GLOSH 分数高 | **随意集 $C$**：按加速倍率 $N$ 进行 RBD 下采样 |

**为什么 HDBSCAN 胜过其他聚类算法？**

| 算法 | 问题 | HDBSCAN 的优势 |
|------|------|--------------|
| K-means | 需要预设 $K$（聚类数），假设球形聚类 | 自动确定聚类数；任意形状 ✓ |
| DBSCAN | 固定 $\epsilon$ 无法处理变密度；需要手动调 $\epsilon$ | 层次化，无需 $\epsilon$；只需设 $m_{clSize}$ ✓ |
| GMM | 假设高斯混合分布，对重尾/不规则分布效果差 | 非参数，无分布假设 ✓ |
| **HDBSCAN** | — | **内置离群点检测**（直接输出随意集 $C$）；对密度变化鲁棒 ✓✓✓ |

---

## 三、两个组件的完整协作逻辑

```
代理策略 π_θ（已在原速数据上训练完毕）
        │
        │  对每帧 o_t，采样 N 次动作块
        ▼
  样本集 { a_j^i[t] }  (N×K 个点)
        │
        │  Gaussian KDE（式 1）
        ▼
  条件密度 p̂(a|o_t)   ← 带宽 h 控制平滑度
        │
        │  Monte Carlo 近似熵（式 2）
        ▼
  每帧标量熵 Ĥ(a_t|o_t)
        │
        │  Isolation Forest 去除异常熵值
        ▼
  清洁熵序列 + 时间索引（二维点集）
        │
        │  HDBSCAN 聚类（式 3-8）
        │     核心距离 → 互达距离 → MST
        │     → Dendrogram → 压缩树 → 稳定性分数
        ▼
  精密集 P（低熵聚类）  ╮
  随意集 C（离群点）    ╯  → RBD 下采样 → 加速示教数据集
                                │
                                ▼
                    训练加速策略（ACT/DP）→ 部署
```

**两者的互补关系**：

- **KDE 的作用**：将策略的"多次采样动作"**平滑**成连续密度，避免离散动作频率统计的稀疏问题；输出稳定的标量熵值
- **HDBSCAN 的作用**：将标量熵序列**自动分段**，无需手动设置熵阈值；利用密度结构直接输出"离群高熵点 = 随意集"，与 RBD 下采样无缝衔接

两者相辅相成：KDE 提供**连续、平滑**的不确定性度量；HDBSCAN 提供**无监督、参数鲁棒**的语义分段。

---

*另见*：[[wiki/sources/2026-04-23 DemoSpeedup]] — 完整方法 Pipeline 与实验结果
