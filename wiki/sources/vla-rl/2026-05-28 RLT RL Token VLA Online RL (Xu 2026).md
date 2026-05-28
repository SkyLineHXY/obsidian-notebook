---
type: source
tags: [VLA, Reinforcement Learning, Online RL, RL Fine-tuning, Physical Intelligence, Representation Learning, Action Chunking]
sources: [raw/assets/papers/VLA+RL/Xu 等 - 2026 - RL Token Bootstrapping Online RL with Vision-Language-Action Models.pdf]
created: 2026-05-28
updated: 2026-05-28
---

# RL Token: Bootstrapping Online RL with Vision-Language-Action Models

**arXiv**: 待确认
**作者**: Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke
**机构**: Physical Intelligence (π.ai)
**项目页**: https://pi.website/research/rlt
**摄取日期**: 2026-05-28
**摄取来源**: 用户添加 PDF + MinerU 转换

[[raw/assets/papers/VLA+RL/Xu 等 - 2026 - RL Token Bootstrapping Online RL with Vision-Language-Action Models.pdf]]
[[raw/sources/papers/VLA+RL/Xu 等 - 2026 - RL Token Bootstrapping Online RL with Vision-Language-Action Models/Xu 等 - 2026 - RL Token Bootstrapping Online RL with Vision-Language-Action Models.md]]

---

## 一句话摘要

RLT 在冻结 π₀.₆ VLA 上添加轻量级 encoder-decoder 提取 "RL token" 作为紧凑状态表征，驱动小型 actor-critic 在数小时内完成真实机器人精密操作任务的在线 RL 微调，成功率从 20% 提升到 65%，速度提升 3×。

---

## 核心背景与动机

### 问题：VLA 真实机器人 RL 微调的两难困境

- **全模型 RL 代价高**：全量 RL 微调十亿参数 VLA（如 RECAP）需大规模数据，难以在数小时内适应特定任务精密阶段
- **轻量方法缺乏 VLA 先验**：HIL-SERL 等系统使用 ResNet 视觉编码器，丢失 VLA 中蕴含的大规模预训练感知与行为先验
- **精密任务的最后一毫米**：VLA 在宏观操作（抓取、搬运）表现良好，但精密接触阶段（螺丝安装、插接）成功率低、速度慢

### 关键洞察

利用冻结 VLA 的内部嵌入，通过**自编码瓶颈**（encoder-decoder transformer）提取紧凑的 "RL token"，作为轻量 actor-critic 的状态输入。RL actor 直接以 VLA 的参考动作块为条件，将在线 RL 转化为对 VLA 已有行为的局部精修（local refinement），而非从零开始的无约束搜索。

---

## 方法：RLT（RL Token）

![[raw/sources/papers/VLA+RL/Xu 等 - 2026 - RL Token Bootstrapping Online RL with Vision-Language-Action Models/images/54ad31ede3cbc899a941d2790e09ef80d0aea1b6cad4c37e297743f5754c5ff3.jpg]]
*Figure 2：RL Token 抽取示意。在冻结的 π₀.₆（SigLIP-400M + Gemma-4B + 860M flow-matching action expert）上插入一个 encoder-decoder transformer，将整段 VLA token 序列压缩为单个 1×2048 的 RL token，供下游轻量 actor-critic 使用。*

### Stage 1：RL Token 提取（自适应 VLA 接口）

在预训练 VLA（π₀.₆）上添加 encoder-decoder transformer $g_\phi$（参数小），以自回归重建 VLA 内部 token 嵌入为目标：

$$
\mathbf{z}_{\mathrm{rl}} = g_\phi\!\left([\mathbf{z}_{1:M},\, \mathbf{e}_{\mathrm{rl}}]\right)_{M+1} \tag{1}
$$

重建损失（VLA 参数冻结）：

$$
\mathcal{L}_{\mathrm{ro}} = \mathbb{E}_\mathcal{D}\left[\sum_{i=1}^M \left\|h_\phi\!\left(d_\phi([\mathbf{z}_{\mathrm{rl}}, \bar{\mathbf{z}}_{1:i-1}])\right)_i - \bar{\mathbf{z}}_i\right\|^2\right] \tag{2}
$$

其中 $\bar{\mathbf{z}}_i = \mathrm{sg}(\mathbf{z}_i)$ 为 stop-gradient，$d_\phi$ 为 decoder transformer。训练完成后 VLA 和 $\phi$ 全部冻结，RL 仅在 $\mathbf{z}_{\mathrm{rl}}$ 上运行。

**设计动机**：RL token 作为信息瓶颈，强制保留任务相关信息，同时维度远小于 VLA 全 token 序列，适合轻量 actor-critic 学习。

### Stage 2：在线 RL 微调（轻量 actor-critic）

**状态**：$\mathbf{x} = (\mathbf{z}_{\mathrm{rl}}, \mathbf{s}^p)$（RL token + 本体感知）

**Critic 训练**（off-policy TD，action chunk 级）：

$$
\mathcal{L}_Q = \mathbb{E}_{(\mathbf{x},\mathbf{a}_{1:C},\mathbf{x}')\sim\mathcal{B}}\left[\left(\hat{Q} - Q_\psi(\mathbf{x},\mathbf{a}_{1:C})\right)^2\right], \quad \hat{Q} = \sum_{t'=1}^C \gamma^{t'-1}r_{t'} + \gamma^C \mathbb{E}_{\mathbf{a}'\sim\pi_\theta}\left[Q_{\psi'}(\mathbf{x}',\mathbf{a}')\right] \tag{3}
$$

**Actor 训练**（以 VLA 参考动作块为条件，KL 正则化）：

$$
\pi_\theta(\mathbf{a}_{1:C}\mid\mathbf{x},\tilde{\mathbf{a}}_{1:C}) = \mathcal{N}(\mu_\theta(\mathbf{x},\tilde{\mathbf{a}}_{1:C}),\,\sigma^2\mathbf{I}) \tag{4}
$$

$$
\mathcal{L}_\pi(\theta) = \mathbb{E}\left[-Q_\psi(\mathbf{x},\mathbf{a}_{1:C}) + \beta\|\mathbf{a}_{1:C} - \tilde{\mathbf{a}}_{1:C}\|_2^2\right], \quad \tilde{\mathbf{a}}_{1:C}\sim\pi_{\mathrm{vla}}(\cdot\mid\mathbf{s},\ell) \tag{5}
$$

$\beta$ 控制 actor 向 VLA 参考动作块的正则化强度。

**参考动作 Dropout**：以一定概率将参考块替换为零，防止 actor 退化为 VLA 的简单复制，强制维持独立动作生成能力。

**动作块子采样**（Subsampling）：以步长 2 从 chunk 内取多个中间状态存入 replay buffer，提升数据效率。

**分工设计**：
- 冻结 VLA → 提供感知理解 + 参考动作建议（宏观先验）
- 轻量 RL actor-critic → 在关键精密阶段做局部精修

### 实现细节（Appendix B）

**RL token 抽取器训练**
- VLA backbone：π₀.₆（SigLIP-400M + Gemma-4B + 860M action expert），全程冻结
- VLA token 嵌入维度：2048，RL token $\mathbf{z}_{\mathrm{rl}}$ 形状 $1\times 2048$
- 训练步数：每任务 **2000–10000** gradient steps（在 1–10 h 单任务演示数据上）
- 可选联合 SFT：损失 $\mathcal{L}_{\mathrm{ro}}(\phi) + \alpha\mathcal{L}_{\mathrm{vla}}(\theta_{\mathrm{vla}})$，仅当 $\alpha > 0$ 时同步微调 VLA
- RL token 输入源：2 个 wrist camera + 1 个 base camera 图像（语言指令在单任务设定下省略）

**Actor-Critic 网络（从零初始化，per-task）**
- 输入：$\mathbf{x} = (\mathbf{z}_{\mathrm{rl}}, \mathbf{s}^p)$；$\mathbf{s}^p$ 视任务而定（螺丝：关节位；扎带/Ethernet/充电器：末端位姿）
- 隐藏层规模：
  - 扎带 / Ethernet / 充电器：**2 层 MLP，hidden=256**
  - 螺丝（最难）：**3 层 MLP，hidden=512**
- Critic：**双 Q ensemble**，target 取两者最小（TD3 风格，Fujimoto 2018）
- Actor：高斯策略，**固定小方差 $\sigma$**；同时接收 $\tilde{\mathbf{a}}_{1:C}$ 作为输入
- 动作维度：14-D per timestep × $C=10$ chunk = **140-D** chunked action
- 控制频率：**50 Hz**

**在线 RL 关键超参**
- Reference-action dropout 概率：**50%**（推理时始终提供 $\tilde{\mathbf{a}}$）
- Action chunk 子采样步长：2 → 每秒机器人数据约产 **25 个训练样本**
- Update-to-data ratio（UTD）：**5**（rollout 与 learning 异步执行）
- Critic / Actor 更新比：**2 : 1**
- 训练规模：每任务 400–1000 episodes，约 **15 min – 5 h 实际机器人数据**
- 奖励信号：人工提供 sparse **+1** 成功信号

**两阶段课程（screw / zip-tie）**
1. 仅 critical-phase 训练（小幅初始随机化）
2. 进入 full-task：前序由 base VLA 执行，进入关键段后切换到 RL 策略
3. 最后一次短 SFT：让 VLA 学会预测「何时交棒」给 RL 策略（用人工 intervention 时机作 label），实现 test-time 自动切换

---

## 实验结果

**任务**：4 个真实机器人精密操作（Critical Phase 评测，50 episodes 各）：
- 螺丝安装（sub-mm 对齐）
- 扎带扣合（毫米精度双臂）
- Ethernet 插接（角度+位置精确对齐）
- 充电器插接（厘米级对齐）

**主要结果（Critical Phase）**：

| 任务            | Base VLA | RLT（ours） |
| ------------- | -------- | --------- |
| 螺丝（成功率）       | 20%      | 65%       |
| 扎带（成功率）       | 35%      | 100%      |
| Ethernet（成功率） | 98%      | 100%      |
| 充电器（成功率）      | 79%      | 100%      |

**速度**：关键阶段执行速度提升最高 3×；某些任务 RLT 策略可超越人类遥操速度。

**全任务评测**（含前序阶段）：
- 螺丝 full-task：8% → 48%
- 扎带 full-task：27% → 86%

**消融分析**：
- w/o RL token（换 ResNet-10）：性能显著下降，说明 VLA 预训练表征的关键作用
- w/o Chunk（单步动作）：成功率下降，信用分配变难
- w/o BC Regularizer（$\beta=0$）：不稳定，策略无法有效约束探索
- w/o Pass-Through（不输入 $\tilde{\mathbf{a}}$）：收敛更慢

**对比基线**：HIL-SERL（在多任务 Ethernet 上失败）、PLD（残差单步策略，精度不足）、DSRL（latent noise 操控）、DAgger（数据蒸馏）。

---

## 局限性与未来工作

- 目前每任务需单独训练，无跨任务共享
- RL token 在当前实现中每个任务固定语言指令，多指令泛化有待验证
- 数小时训练预算限制，复杂多阶段任务整体 success rate 提升幅度受限
- 依赖人类标注成功/失败信号（稀疏奖励）

---

## 与已有方法的关系

- **[[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]]**：同为 π₀.₆ 系 VLA 的 RL 后训练，RECAP 全量微调（离线），RLT 冻结 VLA + 轻量在线 RL；互补的工程路线
- **[[wiki/sources/vla-rl/2026-05-17 GR-RL (Li 2025)]]**：同为"冻结 VLA + 轻量 RL 模块"路线，但 GR-RL 操纵 latent noise，RLT 通过 RL token 提取状态
- **[[wiki/entities/systems/HIL-SERL]]**：直接比较基线，RLT 在 HIL-SERL 框架基础上用 VLA 表征替代 ResNet，并引入参考动作条件化
- **[[wiki/concepts/generative-models/Flow Matching]]**：π₀.₆ 使用 flow-matching action expert，RLT 直接在此 VLA 上构建 RL 接口

---

## 新概念追踪

**首次出现，追踪中**：
- **Reference Action Conditioning（参考动作条件化）**：RL actor 以 VLA 参考动作块为条件并正则化，将 RL 转化为局部精修；仅来源 51
- **Reference Action Dropout**：随机清零参考块防止 actor 退化为复制；仅来源 51
- **Critical Phase Fine-tuning**：仅对任务最精密关键阶段做 RL 微调，避免 RL 干扰已经足够好的前序阶段；仅来源 51

---

## 关联页面

- [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] — 同 VLA 的全量离线 RL 对比
- [[wiki/sources/vla-rl/2026-05-17 GR-RL (Li 2025)]] — 同路线轻量 RL 模块
- [[wiki/entities/systems/HIL-SERL]] — 核心对比基线
- [[wiki/concepts/vla/Vision-Language-Action 模型]] — VLA 背景
