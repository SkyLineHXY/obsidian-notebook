---
type: concept
tags: [Visuomotor RL, Representation Learning, Information Bottleneck, Self-Supervised, Reconstruction, VIB, Encoder Drift]
sources: [raw/sources/papers/VLA+RL/Lei 等 - 2026 - RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning/Lei 等 - 2026 - RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning.md]
created: 2026-05-28
updated: 2026-05-28
---

# Representation Regularization in Visuomotor RL（视觉运动 RL 中的表征正则化）

视觉运动策略（visuomotor policy）含一个端到端训练的视觉编码器 $\phi: o \mapsto z$。当从 IL 切换到 RL 微调时，编码器面临两个矛盾的需求：

- **不冻结** → 让 $\phi$ 适应 RL 阶段的新数据分布（提升上限）
- **不漂移** → 防止 PPO 梯度把潜空间推向退化或塌缩态（保稳定）

**表征正则化（Representation Regularization）** 通过在 IL 阶段额外引入**重建（Reconstruction）+ 变分信息瓶颈（VIB）** 两类辅助损失，在 RL 阶段降权保留，实现"既学又稳"的甜点。该范式在 [[wiki/sources/rl-finetuning/2026-05-28 RL-100 Real-World RL on Diffusion Policy (Lei 2026)|RL-100]] 中被证明是真实机器人 RL 后训练稳定性的关键稳定器。

---

## 1. 核心问题：表征漂移（Representational Drift）

### 1.1 纯 BC 编码器的脆弱性

若仅用噪声预测目标 $\mathcal{L}_{\mathrm{IL}}$ 训练 $\phi$：

$$
\mathcal{L}_{\mathrm{IL}}(\theta) = \mathbb{E}_{(a^{\tau_0}, c_t)\sim\mathcal{D},\,\tau,\,\varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(a^\tau, \tau, c_t)\|_2^2\right]
$$

得到的潜空间 $z = \phi(o, q)$ **仅对当前演示集 $\mathcal{D}_0$ 局部最优**：
- 只保留"区分演示中各 action 所需的特征"
- 大量与任务相关、但与当前演示 action 无关的几何/状态信息被压掉
- $z$ 的分布、范数、几何**无约束**

进入 RL 微调后：
1. 数据分布从 $\mathcal{D}_0$ 漂到 $\mathcal{D}_m$
2. PPO 梯度同时反传到 encoder
3. 编码器极易**塌缩**或**漂移**——原本能用的特征在新分布上失效，actor/critic 跟着崩

### 1.2 两个极端方案均不理想

| 方案 | 优点 | 缺点 |
|------|------|------|
| **完全冻结 encoder** | 表征绝对稳定 | 无法适应新分布，上限受限 |
| **完全联合微调** | 上限高 | 漂移严重，性能崩溃 |

**RL-100 的方案**：联合训练 + Recon/VIB 正则 + RL 阶段降权 = 甜点。

---

## 2. 两类正则项的设计

### 2.1 Reconstruction Regularization（重建正则）

强制编码 $z$ 能重建出原始观测 $(o, q)$：

$$
\mathcal{L}_{\text{recon}} = \beta_{\text{recon}}\!\left(d_{\text{Chamfer}}(\hat{o}, o) + \|\hat{q}-q\|_2^2\right)
\tag{Eq. 4}
$$

其中：
- $o$：原始点云观测
- $q$：本体感知向量（关节角、末端位姿等）
- $\hat{o}, \hat{q}$：从 $z=\phi(o,q)$ 通过解码器重建
- $d_{\text{Chamfer}}$：点云的 Chamfer 距离

**Chamfer 距离**：

$$
d_{\text{Chamfer}}(A, B) = \frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\|a-b\|^2 + \frac{1}{|B|}\sum_{b\in B}\min_{a\in A}\|a-b\|^2
$$

**作用**：编码器**不能只记 action**，必须保留足够多的几何 / 状态信息。当 RL 把分布推到新区域时，重建项保证 $z$ 仍能"描述"那个区域的观测 → 抗漂移。

### 2.2 Variational Information Bottleneck（VIB，变分信息瓶颈）

引入概率编码器 $\phi(z\mid o, s)$（输出高斯分布参数 $\mu, \sigma$），强制其逼近标准正态先验：

$$
\mathcal{L}_{\mathrm{KL}} = \beta_{\mathrm{KL}}\,\mathrm{KL}\!\left(\phi(z\mid o, s)\,\big\|\,\mathcal{N}(0, \mathbf{I})\right)
\tag{Eq. 5}
$$

**作用**：限制 $z$ 的"信息容量"（详见 §3.2 推导），强迫编码器只保留**预测 action 所必需**的信息。潜空间被规范到一个有界、平滑、近高斯的几何——PPO 在其上的梯度更稳定。

### 2.3 完整 IL 目标

$$
\mathcal{L}_{\text{total}}^{\mathrm{IL}} = \mathcal{L}_{\mathrm{IL}} + \mathcal{L}_{\text{recon}} + \mathcal{L}_{\mathrm{KL}}
\tag{Eq. 6}
$$

### 2.4 RL 阶段的衰减策略

> **During RL fine-tuning, we reduce $\beta_{\mathrm{recon}}$ and $\beta_{\mathrm{KL}}$ by a factor of 10.**

| 阶段 | $\beta$ 强度 | 作用 |
|------|-------------|------|
| IL pretraining | $\beta$（原始） | 建立稳定泛化的表征底座 |
| RL fine-tuning | $0.1\,\beta$ | 给策略改进留出空间，但仍维持表征稳定 |

这是典型的"**先约束后放松**"schedule。

---

## 3. 严谨数学推导

### 3.1 重建正则的信息论解释

设编码器为确定性 $z = \phi(o, q)$，加重建项后总损失下界控制了**互信息 $I(z; o, q)$ 的下界**：

$$
\mathcal{L}_{\text{recon}} \geq -\,\beta_{\text{recon}}\,\mathbb{E}[\log p(o, q\mid z)] + \mathrm{const}
$$

由 ELBO 类似的推导：

$$
\log p(o, q) \;\geq\; \mathbb{E}_{q(z\mid o,q)}[\log p(o,q\mid z)] - \mathrm{KL}(q(z\mid o,q)\,\|\,p(z))
$$

第一项即重建似然（Chamfer + L2 是 $\log p$ 的高斯/能量近似）。最大化它等价于**保证 $z$ 含足够多关于 $(o, q)$ 的信息** → 下界 $I(z; (o, q))$。

### 3.2 VIB 的信息瓶颈解释

VIB 损失的本质是**变分上界**互信息 $I(z; o, s)$：

$$
\mathrm{KL}\!\left(\phi(z\mid o,s)\,\|\,r(z)\right) \;\geq\; I(z; o, s) - \mathbb{E}_{o,s}[\mathrm{KL}(\phi(z\mid o,s)\|p(z))]_{r(z)=p(z)} + \mathrm{const}
$$

更精确地（Alemi et al. 2017）：

$$
I(z; o, s) \;\leq\; \mathbb{E}_{(o,s)}\!\left[\mathrm{KL}\!\big(\phi(z\mid o,s)\,\big\|\,r(z)\big)\right]
$$

取变分边缘 $r(z) = \mathcal{N}(0, \mathbf{I})$，即得到 $\mathcal{L}_{\mathrm{KL}}$（系数 $\beta_{\mathrm{KL}}$）。

**因此 VIB 的优化等价于**：

$$
\min_\phi\; -I(z; a) + \beta_{\mathrm{KL}}\cdot I(z; o, s)
$$

（前一项来自 $\mathcal{L}_{\mathrm{IL}}$ 隐含的"$z$ 要能预测 $a$"，后一项即 KL 项）。这是经典**Information Bottleneck（Tishby）目标**：

$$
\min_\phi \;-\,I(z; \text{target}) + \beta\,I(z; \text{input})
$$

直觉：$z$ 是输入到目标的**最小充分统计量**。

### 3.3 Re-parameterization Trick

VIB 实现需要可微采样：

$$
z = \mu_\phi(o, s) + \sigma_\phi(o, s)\odot\varepsilon,\quad \varepsilon\sim\mathcal{N}(0, \mathbf{I})
$$

KL 项对高斯有闭式：

$$
\mathrm{KL}(\mathcal{N}(\mu, \sigma^2\mathbf{I})\,\|\,\mathcal{N}(0,\mathbf{I})) = \frac{1}{2}\sum_j\!\left(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right)
$$

### 3.4 为何这两个正则在 RL 阶段降权？

记 RL 阶段总损失：

$$
\mathcal{L}^{\mathrm{RL}}_{\text{total}} = \mathcal{L}_{\mathrm{RL}} + \beta'_{\text{recon}}\mathcal{L}_{\text{recon}} + \beta'_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}}
$$

若 $\beta'$ 不降权：
- 编码器优化几乎被两个正则项主导，$z$ 难以为 RL 改进腾出表征空间
- 策略改进受阻，PPO 收益微弱

若 $\beta' = 0$：
- IL 阶段习得的表征几何快速漂移
- 演示之外区域的 $z$ 分布发散，critic / actor 崩

降权 10× 是经验上的甜点：**保留正则梯度方向，但允许 RL 主导优化**。

---

## 4. 实验证据（RL-100 Adroit Door Ablation）

| 配置 | 1.0M env step 成功率 | 稳定性 |
|------|---------------------|--------|
| **w/ ReconVIB (ours)** | **1.00** | 最稳，confidence interval 最窄 |
| w/o ReconVIB | 0.90 | 振荡明显 |
| Fix encoder | 0.95 | 中等，上限受限 |

**论文原文结论**：

> "Joint policy–representation adaptation **mitigates representational drift** and improves sample efficiency."

含义：联合训练 encoder + Recon/VIB 正则 = 既能适应新分布（不冻结），又不会漂移（有正则锚定）。

---

## 5. 设计要点 & 选型建议

| 维度 | 选项 | 何时选 |
|------|------|--------|
| 重建目标 | 点云 Chamfer | 输入为点云 / 3D（如 DP3 backbone） |
| 重建目标 | RGB 像素 MSE / SSIM | 输入为图像 |
| 重建目标 | MAE-style masked reconstruction | 高维输入，提升语义抽象 |
| Bottleneck 形式 | VIB（高斯 KL） | 经典选择，闭式 KL，易实现 |
| Bottleneck 形式 | InfoNCE / contrastive | 不需解码器，但更难调 |
| $\beta_{\mathrm{KL}}$ 调度 | 余弦 / 线性 | 训练初期小 → 后期大，避免 posterior collapse |
| RL 阶段降权倍率 | 10× | RL-100 经验值；可在 5–20× 间调 |
| Encoder 是否冻结 | 不冻结 | 配合正则可达最高性能 |

---

## 6. 历史脉络与同类方法

| 工作 | 表征正则手段 |
|------|-------------|
| **VIB**（Alemi et al. 2017） | 信息瓶颈的变分实现，本概念的理论基石 |
| **β-VAE**（Higgins et al. 2017） | 同款 KL 项，目标在 disentanglement |
| **DrQ / DrQ-v2**（Yarats 2021） | 数据增强 + 表征对齐（不重建） |
| **R3M / VC-1**（Nair 2022 / Majumdar 2023） | 预训练通用视觉表征，冻结使用 |
| **DP3**（Ze 2024） | 3D 点云策略 backbone（RL-100 即用其编码器 + Recon） |
| **RL-100**（Lei 2026） | Recon + VIB 联合，RL 阶段降权 10× |

---

## 7. 在知识库中的出现

| 来源 | 角色 |
|------|------|
| [[wiki/sources/rl-finetuning/2026-05-28 RL-100 Real-World RL on Diffusion Policy (Lei 2026)\|RL-100 (Lei 2026)]] | 主体：将 Recon + VIB 系统化用于真实机器人 RL 后训练，提供消融证据 |

**待入库相关方法**（论文中提到但尚未单独建页）：
- **DP3**（Ze 2024，已被 RL-100 采纳为 3D 编码器 backbone）
- **R3M**（Nair 2022）/ **MoCo**（He 2020，作为对比基线提到）

---

## 8. 关联页面

- [[wiki/concepts/rl/Iterative Offline RL]] — RL-100 同篇的策略改进算法
- [[wiki/sources/rl-finetuning/2026-05-28 RL-100 Real-World RL on Diffusion Policy (Lei 2026)]] — 来源
- [[wiki/concepts/rl/DPPO]] — 同类 PPO 微调扩散策略方法（未含表征正则）
- [[wiki/analyses/RL微调生成模型的技术挑战与解决方案]] — 方法背景
