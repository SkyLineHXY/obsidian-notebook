---
type: source
tags: [ImitationLearning, DemonstrationAcceleration, ActionEntropy, ACT, DiffusionPolicy, DataCuration, KDE, HDBSCAN]
sources: [raw/assets/papers/IL(Imitation Learning)/Guo 等 - 2025 - DemoSpeedup Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration.pdf]
created: 2026-04-23
updated: 2026-04-24
---

[[raw/sources/papers/IL(Imitation Learning)/Guo 等 - 2025 - DemoSpeedup Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration/Guo 等 - 2025 - DemoSpeedup Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration.md]]

# DemoSpeedup（Guo et al., 2025）

**标题**：DemoSpeedup: Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration  
**机构**：Shanghai Qi Zhi Institute / Tsinghua IIIS / Shanghai AI Lab / UESTC  
**关键词**：Imitation Learning, Manipulation, Demonstration Acceleration  
**项目主页**：https://demospeedup.github.io/

---

## 核心问题

人类 Teleoperation 采集的示教数据天生偏慢（VR/kinematic-teaching 的视觉受限、触觉缺失、设备延迟等），导致行为克隆（BC）策略执行速度也偏慢。

- **Test-time acceleration（测试时加速）**：暴力下采样动作块，但 2× 时性能下降明显（分布偏移），不够理想。
- **DemoSpeedup**：在**训练数据层面**对示教轨迹进行自监督加速，从根本上消除分布偏移。

---

## 核心方法：四步 Pipeline

### 1. 代理策略训练

在原速示教数据上训练任意生成策略（[[wiki/concepts/imitation-learning/ACT]] 或 [[wiki/concepts/generative-models/Diffusion Policy]]）作为代理策略，**仅用于动作熵估计，不用于最终部署**。

### 2. 逐帧动作熵估计（Gaussian KDE）

对每帧观测 $o_t$，从代理策略采样 $N$ 个动作块，用 Gaussian KDE 估计条件概率密度：

$$\hat{p}(a_t \mid o_t) = \frac{1}{NKh} \sum_{j=t-K+1}^{t} \sum_{i=1}^{N} \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{(a_t - a_j^i[t])^2}{2h^2}\right)$$

再计算条件动作熵：

$$\hat{H}(a_t \mid o_t) = -\sum_{j=t-K+1}^{t} \sum_{i=1}^{N} \hat{p}(a_j^i[t] \mid o_t) \log \hat{p}(a_j^i[t] \mid o_t)$$

**核心洞见**：
- **低熵帧** → 策略行为一致 → 精密操作（抓取、插入）→ **低速保留**
- **高熵帧** → 策略行为多样 → 随意运动（空中移动）→ **高速下采样**

### 3. HDBSCAN 密度聚类分段

先用 Isolation Forest 去除异常熵值，再用层次密度聚类 HDBSCAN 将熵点分为：
- 精密集 $P$（Precision Set）：均值熵 < 0 的低熵聚类
- 随意集 $C$（Casualness Set）：高熵点（标记为离群值）

### 4. Replicate-Before-Downsample（RBD）策略

以 $N\times$ 加速率处理随意集时，将动作块复制 $N$ 份，第 $i$ 份以偏移 $i$ 帧下采样 $N\times$，**保留原始状态多样性，避免信息损失**。

---

## 实验结果

### 仿真（ALOHA + BiGym，11 任务）

| 方法 | 平均成功率 | 平均加速 |
|------|-----------|---------|
| ACT（原速） | 77% | 1.0× |
| ACT-2×（测试时加速） | 69% | 1.7× |
| **ACT + DemoSpeedup** | **82%** | **2.1×** |
| DP（原速） | 55% | 1.0× |
| DP-2×（测试时加速） | 45% | 1.6× |
| **DP + DemoSpeedup** | **59%** | **1.9×** |

### 真实机器人（Galaxea R1 人形机器人，5 任务 + 1 变体）

- **Sort**：ACT+DemoSpeedup 用时 20.38s vs ACT 56.78s → **2.78× 加速**，成功率持平
- **Conveyor Fast**（2× 速传送带）：DP+DemoSpeedup **27/30** 成功，原版 DP 仅 7/30
- **最大加速 3×**，多任务成功率持平或提升

### 消融实验（关键设计验证）

| 消融对象 | ACT 成功率 | DP 成功率 |
|---------|-----------|---------|
| DemoSpeedup（完整） | 56% | 52% |
| w/o RBD 策略 | 29% | 26% |
| w/o 几何一致性块长 | 31% | 34% |
| w/o 高精度控制器 | 53% | 41% |

---

## 关键洞见

1. **成功率有时反而提升**：加速降低决策时域 → 减少复合误差（Compounding Error）
2. **RBD 策略是关键**：消融后 ACT 成功率从 56% 跌至 29%
3. **适用范围广**：对 ACT（CVAE 采样）和 DP（去噪多次采样）均有效
4. **真实数据往往比仿真数据慢**，DemoSpeedup 在真实场景增益更大

---

## 局限性

- 期望加速率需**手动设定**（不同数据集采集速度有差异）
- DP 推理延迟会引入执行停顿，影响加速效果（可用蒸馏或流策略解决）
- 偶发性能下降（原速与加速示教间的动力学不匹配）

---

## Related Concepts & Entities

- [[wiki/concepts/imitation-learning/ACT]] — DemoSpeedup 以 ACT 为代理策略和最终策略之一；该论文为 ACT 新增来源
- [[wiki/concepts/generative-models/Diffusion Policy]] — DemoSpeedup 以 DP 为代理策略和最终策略之一；该论文为 DP 新增来源
- [[wiki/concepts/vla/Vision-Language-Action 模型]] — 大规模 VLA 数据集同样面临慢速示教问题，DemoSpeedup 可直接延伸

## 深度分析

- [[wiki/analyses/DemoSpeedup 逐帧熵估计与HDBSCAN聚类数学推导]] — Gaussian KDE 密度估计、条件熵计算、HDBSCAN 从核心距离到聚类稳定性的完整数学推导链
