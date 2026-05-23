---
type: comparison
tags: [VLA, RL-Finetuning, Vision-Language-Action, Online-RL, Offline-RL, Post-Training, Flow-Matching, Autoregressive, PPO, On-Policy-Distillation]
sources:
  - raw/sources/papers/VLA+RL/Intelligence 等 - 2025 - π₀.₆ a VLA That Learns From Experience.pdf
  - raw/sources/papers/VLA+RL/Chen 等 - 2026 - πRL Online RL Fine-tuning for Flow-based VLAs.pdf
  - raw/assets/papers/VLA+RL/Guo - 2025 - iRe-VLA.pdf
  - raw/assets/papers/VLA+RL/Lu - 2025 - VLA-RL.pdf
  - raw/assets/papers/VLA+RL/Li - 2025 - GR-RL.pdf
  - raw/assets/papers/VLA+RL/Pan - 2026 - SA-VLA.pdf
  - raw/assets/papers/VLA+RL/Zhong - 2026 - VLA-OPD.pdf
created: 2026-05-17
updated: 2026-05-17
---

# VLA RL 微调方法对比

> **覆盖范围**：策略骨干为**大型 VLA（携带 LLM 或 VLM）**的 RL 后训练方法族，共 7 项工作：RECAP/π₀.₆、πRL、iRe-VLA、VLA-RL、GR-RL、SA-VLA、VLA-OPD。纯生成策略（Diffusion / Flow，无 LLM 骨干）的 RL 微调参见 [[wiki/comparisons/RL 微调表达性策略方法对比]]。

---

## 为什么 VLA RL 是独立问题

VLA（Vision-Language-Action）模型将大型语言/视觉语言模型作为策略骨干，带来一套特有挑战，与纯 Diffusion / Flow 策略 RL 有结构性差异：

| 维度 | 纯生成策略（Diffusion/Flow） | VLA（大模型骨干） |
|---|---|---|
| **参数规模** | 数百 M | 7B–70B |
| **log-prob 计算** | 迭代 ODE，不直接可算（主要挑战） | AR-VLA 有 softmax logits（相对容易），Flow-VLA 同样困难 |
| **训练稳定性来源** | 主要来自 ODE 离散化误差 | 额外来自大模型 RL 的灾难性遗忘 |
| **数据异构性** | 通常单一训练集 | 演示 + 自主 rollout + 人工干预多源混合 |
| **泛化 vs 专精张力** | 任务特化 | 需平衡通用语言理解与特定任务精度 |
| **奖励信号** | 仿真器直接提供 | 稀疏、含噪、需要设计或学习 |

---

## 两条 VLA 架构路线

### 路线 A：自回归 VLA（AR-VLA）

**代表**：iRe-VLA、VLA-RL、VLA-OPD；GR-RL（通用 VLA 含 AR 架构）

- 动作被量化为离散 token，输出分布通过 softmax logits 得到
- **Log-probability 可由 token logits 直接计算** → RL 算法可直接套用 PPO / REINFORCE
- 主要挑战：（1）灾难性遗忘——RL 破坏预训练语言理解能力；（2）量化误差——离散 token 导致连续控制精度受限

### 路线 B：Flow-based VLA

**代表**：RECAP/π₀.₆、πRL、SA-VLA

- 动作头使用 Flow Matching 参数化（连续、高精度）
- **Log-probability 不可直接计算** → 需要专门解法（同纯 Flow 策略 RL 难题，见 [[wiki/comparisons/RL 微调表达性策略方法对比]]）
- 主要挑战：在 flow 动力学层面适配 RL，同时保持大模型预训练能力

---

## 方法全景对比

| 方法                                                               | VLA 架构              | RL 范式                 | Log-prob 方案                         | 核心创新                                 | 奖励类型        | 规模                 |
| ---------------------------------------------------------------- | ------------------- | --------------------- | ----------------------------------- | ------------------------------------ | ----------- | ------------------ |
| [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP\|RECAP/π₀.₆]]        | Flow VLA            | 迭代离线 RL               | **不做 PG**，advantage conditioning    | Advantage 作为前缀 token 条件化 VLA 生成      | 稀疏二值 + 价值函数 | 真实世界，大规模多任务        |
| [[wiki/sources/vla-rl/2026-04-24 πRL\|πRL]]                      | Flow VLA (π₀/π₀.₅)  | 在线 RL (PPO)           | Flow-Noise（噪声注入）/ Flow-SDE（ODE→SDE） | 两种等价方案使 flow VLA 可做 PPO              | 仿真稀疏        | 仿真多基准              |
| [[wiki/sources/vla-rl/2026-05-17 SA-VLA (Pan 2026)\|SA-VLA]]     | Flow VLA            | 在线 RL                 | 类似 flow 策略 RL                       | 空间对齐 + 几何进度奖励 + SCAN 退火探索            | 稠密（几何进度）    | 仿真多物体              |
| [[wiki/sources/vla-rl/2026-05-17 iRe-VLA (Guo 2025)\|iRe-VLA]]   | AR-VLA              | 迭代 RL + SFT 交替        | AR logits                           | 交替 RL 探索 + SFT 稳定的迭代框架               | 稀疏          | 仿真 + 真实臂           |
| [[wiki/sources/vla-rl/2026-05-17 VLA-RL (Lu 2025)\|VLA-RL]]      | AR-VLA (OpenVLA-7B) | 在线 RL (PPO/REINFORCE) | AR logits                           | VLM 过程奖励模型（RPRM）稠密化奖励                | 稠密（VLM 生成）  | 仿真 LIBERO-40       |
| [[wiki/sources/vla-rl/2026-05-17 GR-RL (Li 2025)\|GR-RL]]        | AR / 通用 VLA         | 离线 RL 过滤 + 在线精调       | Q-function 过滤（不需 log-prob）          | Q 值作任务进度函数，形态对称增强，潜空间噪声预测            | 稀疏（Q 隐式稠密化） | 真实机器人（极限精度）        |
| [[wiki/sources/vla-rl/2026-05-17 VLA-OPD (Zhong 2026)\|VLA-OPD]] | AR-VLA              | On-policy 蒸馏（非 RL 奖励） | Reverse-KL                          | 教师蒸馏在 policy-induced 状态上，桥接 SFT 与 RL | 无（教师替代奖励）   | 仿真 LIBERO/RoboTwin |

---

## 核心技术挑战与解法矩阵

### 挑战 1：如何为 Flow-based VLA 提供 log-prob / PG 信号

| 解法类型              | 方法               | 机制                                                                           |
| ----------------- | ---------------- | ---------------------------------------------------------------------------- |
| **完全回避 PG**       | RECAP            | Advantage conditioning：把 advantage 二值化后作为 prefix token，让 VLA 自回归条件化生成；无需策略梯度 |
| **噪声注入 Markov 化** | πRL (Flow-Noise) | 与 ReinFlow 类似，在 flow VLA 去噪步注入可学习噪声 → 封闭 log-prob → PPO                      |
| **ODE → SDE 转换**  | πRL (Flow-SDE)   | Rectified Flow ODE 转等价 SDE，离散化后每步为高斯分布 → log-prob 可算                         |

### 挑战 2：如何处理稀疏奖励（信用分配）

| 解法                         | 方法      | 机制                                                                                |
| -------------------------- | ------- | --------------------------------------------------------------------------------- |
| **VLM 过程奖励模型（PRM）**        | VLA-RL  | 微调 VLM 为 RPRM，在任务分段上自动产生伪稠密奖励 $\{r_t\}_{t=0}^T$                                   |
| **离线 Q-function 隐式稠密化**    | GR-RL   | 先用离线 RL 学 $Q^\pi(s,a)$，高 Q 值转移保留（阶段 1），隐式过滤次优数据                                   |
| **几何进度奖励**                 | SA-VLA  | $r_t = \alpha r_{\text{task}} + \beta r_{\text{geo}}(s_t, g)$，末端执行器与目标的几何对齐作为稠密信号 |
| **Advantage conditioning** | RECAP   | 二值化 advantage 作为生成条件，将价值函数信息注入 token 分布，无需逐步奖励信号                                  |
| **教师替代奖励**                 | VLA-OPD | 用专家教师对学生 token 提供稠密监督，完全绕开稀疏环境奖励                                                  |
| **稀疏 + 迭代稳定**              | iRe-VLA | 直接使用稀疏奖励，通过 RL+SFT 交替循环减缓训练崩溃                                                     |

### 挑战 3：训练稳定性与灾难性遗忘

| 解法                                | 方法      | 机制                                                      |
| --------------------------------- | ------- | ------------------------------------------------------- |
| **迭代 RL + SFT**                   | iRe-VLA | RL 探索 → SFT 拉回分布稳定性，交替循环                                |
| **Reverse-KL On-policy 蒸馏**       | VLA-OPD | 在策略诱导状态上学习，自然对齐训练/部署分布；KL 约束防遗忘                         |
| **空间对齐辅助损失**                      | SA-VLA  | 辅助对比目标 $\mathcal{L}_{\text{spatial}}$ 保持空间表示结构不被 RL 破坏  |
| **离线预过滤**                         | GR-RL   | 阶段 1 离线过滤次优数据，减少后续 RL 对脏数据的依赖，降低 RL 收敛风险                |
| **大规模 offline base + 迭代 rollout** | RECAP   | π₀.₆ 三段式：offline base → 任务特化 → 迭代 rollout 改进，稳定性通过数据量保障 |

---

## 方法演化谱系与关系图

```
          AR-VLA 路线                              Flow-VLA 路线
                │                                       │
        iRe-VLA (2025-01)                  RECAP / π₀.₆ (2025-11)
     [RL+SFT 交替迭代，早期探索]        [Advantage Conditioning，大规模真实部署]
                │                                       │
        VLA-RL (2025-05)                       πRL (2026)
     [PRM 稠密化 + Scaling Laws]         [Flow-Noise / Flow-SDE，PPO on Flow-VLA]
                │                                       │
        VLA-OPD (2026-03)                    SA-VLA (2026-01)
     [Reverse-KL 蒸馏，SFT-RL 统一]    [空间归纳偏置保护，SCAN 退火探索]

          GR-RL (2025-12)：三阶段流水线（离线 Q 过滤→对称增强→在线精调）
                      → 聚焦极限精度，与两条路线均有关联
```

---

## 四大设计哲学分歧

### 1. PG vs. Conditioning：如何注入 RL 信号

**RECAP 立场**：PPO 等 PG 方法难以扩展到大型 flow-based VLA，提出 **advantage conditioning** 作为替代——advantage 不是梯度信号，而是生成条件 prefix token。代价：依赖精准的价值函数，无法直接利用 on-policy 探索梯度。

**πRL / VLA-RL 立场**：PPO 对 VLA 是可行的，只需解决 log-prob 计算问题。πRL 通过 Flow-Noise/Flow-SDE 解决；VLA-RL 直接利用 AR logits。实证：πRL 在 LIBERO 等基准上 +29–31%，支持该立场。

### 2. 在线 vs. 离线：数据来源和计算成本

- **RECAP、πRL、VLA-RL、SA-VLA、iRe-VLA**：依赖**在线交互**（仿真 rollout 或真实机器人），需要持续环境访问权限
- **GR-RL 阶段 1**：**完全离线**，仅需历史演示数据做 Q-function 学习
- **VLA-OPD**：策略自生成轨迹（on-policy），不需要稀疏环境奖励，但仍需运行环境（用于采集学生轨迹）

### 3. 奖励来源：环境 vs. VLM vs. 几何 vs. 教师

| 来源 | 方法 | 优点 | 缺点 |
|---|---|---|---|
| 仿真器稀疏二值奖励 | πRL, iRe-VLA | 简单，无需额外设计 | 信用分配困难，训练慢 |
| VLM 过程奖励（RPRM） | VLA-RL | 自动稠密化，可扩展 | 需要额外微调 VLM；奖励准确性受限 |
| 几何进度奖励 | SA-VLA | 与任务强相关，稳定 | 需要明确的任务几何结构 |
| Q-function 隐式过滤 | GR-RL | 无需在线奖励，抗噪声 | 需要演示数据中有足够多样的 Q-value 分布 |
| 教师蒸馏（无奖励） | VLA-OPD | 完全不依赖奖励设计 | 受限于教师模型质量 |

### 4. 通用性 vs. 专精性

- **RECAP** 和 **VLA-RL** 追求**通用性**：多任务、多场景、scaling laws（VLA-RL 初步验证机器人领域的 inference scaling laws）
- **GR-RL** 追求**极限精度**：穿鞋带（毫米级精度 + 长时域），首次用学习型策略实现 83.3% 成功率
- **SA-VLA** 追求**空间鲁棒性**：RL 微调后仍能零样本泛化到新位置、新物体排布
- **VLA-OPD** 追求**样本效率**：2–3× 样本数减少达到相同性能，同时缓解灾难性遗忘

---

## KL 目标的深度分析（VLA-OPD 独特贡献）

VLA-OPD 对三种常见对齐目标做了系统分析，揭示了 VLA 训练不稳定的根源：

| 目标 | 形式 | 行为 | 问题 |
|---|---|---|---|
| Forward-KL（模式覆盖） | $\text{KL}(\pi_{\text{teacher}} \| \pi_\theta)$ | 学生尝试覆盖教师所有模式 | **熵爆炸**：动作分布过于分散 |
| Hard-CE（硬复制） | $-\log \pi_\theta(a^* \| s)$ | 强迫学生精确复制最可能动作 | **熵崩溃**：过早收敛，失去多样性 |
| **Reverse-KL**（模式搜索） | $\text{KL}(\pi_\theta \| \pi_{\text{teacher}})$ | 学生选择性覆盖教师高置信区域 | **有界**：自动过滤教师不确定区域 |

$$\text{KL}(\pi_\theta \| \pi_{\text{teacher}}) = \sum_a \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_{\text{teacher}}(a|s)}$$

当 $\pi_{\text{teacher}}(a|s) \to 0$ 时，若 $\pi_\theta(a|s) > 0$，该项趋向 $+\infty$，自然惩罚在教师无信心处生成动作。这一分析对所有 VLA 对齐训练都有参考价值。

---

## 定量性能对比

| 方法 | 基准 | 关键指标 |
|---|---|---|
| RECAP/π₀.₆ | 真实机器人（叠衣、咖啡、装箱）| 任务吞吐翻倍，失败率减半；连续运行 13 小时 |
| πRL | LIBERO / ManiSkill / MetaWorld / CALVIN | π₀ 平均 **+29.2%**；π₀.₅ 平均 **+31.0%** 成功率 |
| iRe-VLA | Franka Kitchen, RoboSuite + 真实臂 | 相比纯 RL 更稳定；相比纯 SFT 在 OOD 更泛化 |
| VLA-RL | LIBERO-40（40 任务操控） | 超越有监督 SOTA **+4.5%**；匹配商用 π₀-FAST |
| GR-RL | 穿鞋带（真实机器人）| 成功率 **83.3%**（据称首次用学习型策略实现） |
| SA-VLA | 多物体 / 杂乱场景操控 | 空间泛化显著提升；SCAN 探索贡献最大 |
| VLA-OPD | LIBERO, RoboTwin 2.0 | 样本效率提升 **2–3×**；灾难性遗忘明显减少 |

---

## 实用选型建议

| 场景                       | 推荐方法        | 核心理由                                  |
| ------------------------ | ----------- | ------------------------------------- |
| 大规模多任务 Flow-VLA + 真实部署迭代 | **RECAP**   | 唯一经真实世界大规模验证的 Flow-VLA RL 路线          |
| Flow-based VLA + 仿真在线 RL | **πRL**     | +29–31% 成功率；Flow-Noise 实现相对简单         |
| Flow VLA + 空间泛化/新位置 OOD  | **SA-VLA**  | 专门防止 RL 微调破坏空间归纳偏置                    |
| AR-VLA + 通用性 / Scaling   | **VLA-RL**  | PRM 稠密化奖励 + scaling laws；OpenVLA 开源可用 |
| AR-VLA + 极限精度 / 长时域灵巧任务  | **GR-RL**   | 三阶段流水线，Q 过滤 + 对称增强 + 真机精调             |
| AR-VLA + 样本效率 + 防遗忘      | **VLA-OPD** | Reverse-KL 蒸馏，2–3× 样本减少，同时保留预训练能力     |
| 快速原型，AR-VLA，计算资源有限       | **iRe-VLA** | 迭代框架简单，无需大规模系统和精细奖励设计                 |

---

## 相关页面

- 策略级 RL 微调（无 LLM 骨干）：[[wiki/comparisons/RL 微调表达性策略方法对比]]
- VLA 模型综述：[[wiki/concepts/vla/Vision-Language-Action 模型]]
- RL 核心概念：[[wiki/concepts/rl/RECAP]]，[[wiki/concepts/rl/DPPO]]，[[wiki/concepts/rl/Offline 强化学习]]
- 生成模型基础：[[wiki/concepts/generative-models/Flow Matching]]
- 工程框架：[[wiki/entities/frameworks/RLinf]]
- 源页面：[[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]]，[[wiki/sources/vla-rl/2026-04-24 πRL]]，[[wiki/sources/vla-rl/2026-05-17 iRe-VLA (Guo 2025)]]，[[wiki/sources/vla-rl/2026-05-17 VLA-RL (Lu 2025)]]，[[wiki/sources/vla-rl/2026-05-17 GR-RL (Li 2025)]]，[[wiki/sources/vla-rl/2026-05-17 SA-VLA (Pan 2026)]]，[[wiki/sources/vla-rl/2026-05-17 VLA-OPD (Zhong 2026)]]
