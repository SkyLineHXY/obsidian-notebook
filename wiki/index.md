# Wiki Index

> **Content-oriented catalog.** Every wiki page is listed here with a one-line summary. The LLM reads this first when answering any query, then drills into individual pages.
> Updated on every ingest.

---

## Stats
- **Sources**: 46 | **Entities**: 12 | **Concepts**: 13 | **Comparisons**: 2 | **Analyses**: 9
- **Last updated**: 2026-05-25 (Ingest 4 篇待处理论文：FPO++ Yi 2026、LaST-R1 Chen 2026、π0-FPO Lyu 2025、VLAC Zhai 2025；CFM Loss Ratio 概念达 ≥3 来源阈值，待升级为 Concept 页)

---

## Sources by Theme

### Agent Systems (1)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 26 | [[wiki/sources/agent-systems/2026-05-16 ARIS]] | raw/sources/github/wanshuiyin...ARIS...md | 2026-05-16 |

### VLA + RL (10)
> 策略骨干为携带 LLM/VLM 的 VLA 模型，RL 作用于大模型级别的策略后训练。
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 7 | [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] | raw/sources/papers/VLA+RL/Intelligence - 2025 - π₀.₆...pdf | 2026-04-19 |
| 21 | [[wiki/sources/vla-rl/2026-04-24 πRL]] | raw/sources/papers/VLA+RL/Chen - 2026 - πRL | 2026-04-24 |
| 33 | [[wiki/sources/vla-rl/2026-05-17 iRe-VLA (Guo 2025)]] | raw/assets/papers/VLA+RL/Guo - 2025 - iRe-VLA.pdf | 2026-05-17 |
| 34 | [[wiki/sources/vla-rl/2026-05-17 VLA-RL (Lu 2025)]] | raw/assets/papers/VLA+RL/Lu - 2025 - VLA-RL.pdf | 2026-05-17 |
| 35 | [[wiki/sources/vla-rl/2026-05-17 GR-RL (Li 2025)]] | raw/assets/papers/VLA+RL/Li - 2025 - GR-RL.pdf | 2026-05-17 |
| 36 | [[wiki/sources/vla-rl/2026-05-17 SA-VLA (Pan 2026)]] | raw/assets/papers/VLA+RL/Pan - 2026 - SA-VLA.pdf | 2026-05-17 |
| 37 | [[wiki/sources/vla-rl/2026-05-17 VLA-OPD (Zhong 2026)]] | raw/assets/papers/VLA+RL/Zhong - 2026 - VLA-OPD.pdf | 2026-05-17 |
| 44 | [[wiki/sources/vla-rl/2026-05-25 LaST-R1 (Chen 2026)]] | raw/sources/papers/VLA+RL/Chen 等 - 2026 - LaST-R1.../...md | 2026-05-25 |
| 45 | [[wiki/sources/vla-rl/2026-05-25 π0-FPO RFT Flow-VLA (Lyu 2025)]] | raw/sources/papers/VLA+RL/Lyu 等 - 2025 - Reinforcement Fine-Tuning.../...md | 2026-05-25 |
| 46 | [[wiki/sources/vla-rl/2026-05-25 VLAC (Zhai 2025)]] | raw/sources/papers/VLA+RL/Zhai 等 - 2025 - VLA-Critic.../...md | 2026-05-25 |

### 生成模型 + RL（Diffusion / Flow）(10)
> 策略骨干为扩散模型或 Flow Matching（无 LLM 骨干），RL 解决连续去噪链的策略优化问题。HIL-SERL 为传统 SAC 基线系统，纳入同类作比较。
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 4 | [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] | raw/sources/papers/VLA+RL/Zhang - 2026 - ReinFlow...pdf | 2026-04-18 |
| 6 | [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]] | raw/sources/papers/VLA+RL/Park - 2025 - Flow Q-Learning.pdf | 2026-04-19 |
| 8 | [[wiki/sources/rl-finetuning/2026-04-19 DPPO]] | raw/sources/papers/VLA+RL/Ren - 2024 - DPPO.pdf | 2026-04-19 |
| 22 | [[wiki/sources/rl-finetuning/2026-04-24 HIL-SERL]] | raw/sources/papers/VLA+RL/Luo - 2025 - HIL-SERL | 2026-04-24 |
| 28 | [[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]] | raw/assets/papers/VLA+RL/Gao - 2026 - FlowRL...pdf | 2026-05-17 |
| 29 | [[wiki/sources/rl-finetuning/2026-05-17 FPO Flow Matching Policy Gradients (McAllister 2025)]] | raw/assets/papers/VLA+RL/McAllister - 2025 - FPO...pdf | 2026-05-17 |
| 30 | [[wiki/sources/rl-finetuning/2026-05-17 WarmPrior (Kang 2026)]] | raw/assets/papers/VLA+RL/Kang - 2026 - WarmPrior...pdf | 2026-05-17 |
| 31 | [[wiki/sources/rl-finetuning/2026-05-17 OFQL (Nguyen 2025)]] | raw/assets/papers/VLA+RL/Nguyen - 2025 - OFQL...pdf | 2026-05-17 |
| 32 | [[wiki/sources/rl-finetuning/2026-05-17 FAN (Lee 2026)]] | raw/assets/papers/VLA+RL/Lee - 2026 - FAN...pdf | 2026-05-17 |
| 43 | [[wiki/sources/rl-finetuning/2026-05-25 FPO++ (Yi 2026)]] | raw/sources/papers/VLA+RL/Yi 等 - 2026 - Flow Policy Gradients.../...md | 2026-05-25 |

### Generative Models (2)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 14 | [[wiki/sources/generative/2026-04-19 Flow Matching (Lipman 2023)]] | raw/sources/papers/Generative Model/Lipman - 2023 - Flow Matching...pdf | 2026-04-19 |
| 15 | [[wiki/sources/generative/2026-04-21 DDPM (Ho 2020)]] | raw/sources/papers/Generative Model/Ho - 2020 - DDPM.pdf | 2026-04-21 |

### Imitation Learning (3)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 3 | [[wiki/sources/imitation-learning/2026-04-18 复现Diffusion-policy模型]] | raw/sources/blogs/复现Diffusion-policy模型.md | 2026-04-18 |
| 5 | [[wiki/sources/imitation-learning/2026-04-19 Diffusion Policy (Chi 2024)]] | raw/sources/papers/IL/Chi - 2024 - Diffusion Policy...pdf | 2026-04-19 |
| 16 | [[wiki/sources/imitation-learning/2026-04-21 ACT - ALOHA (Zhao 2023)]] | raw/sources/papers/IL/Zhao - 2023 - Bimanual...pdf | 2026-04-21 |

### VLA (3)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 18 | [[wiki/sources/vla/2026-04-23 VLASH]] | raw/assets/papers/VLA/Tang - 2025 - VLASH...pdf | 2026-04-23 |
| 25 | [[wiki/sources/vla/2026-04-29 π₀.₇]] | raw/assets/papers/VLA/Intelligence - 2026 - π₀.₇...pdf | 2026-04-29 |
| 27 | [[wiki/sources/vla/2026-05-17 BayesianVLA (Lian 2026)]] | https://hf.co/papers/2601.15197 | 2026-05-17 |

### Frameworks (4)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 9 | [[wiki/sources/frameworks/2026-04-19 LeRobot]] | raw/sources/github/huggingfacelerobot...md | 2026-04-19 |
| 10 | [[wiki/sources/frameworks/2026-04-19 RLinf]] | raw/sources/github/RLinf...README.md | 2026-04-19 |
| 11 | [[wiki/sources/frameworks/2026-04-19 StarVLA]] | raw/sources/github/starVLA...README.md | 2026-04-19 |
| 19 | [[wiki/sources/frameworks/2026-04-24 RLinf文档]] | raw/sources/blogs/RLinf 文档.md | 2026-04-24 |

### Data Collection (2)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 23 | [[wiki/sources/data-collection/2026-04-24 UMI (Chi 2024)]] | raw/sources/papers/IL/Chi - 2024 - UMI... | 2026-04-24 |
| 24 | [[wiki/sources/data-collection/2026-04-24 FastUMI (Zhaxizhuoma 2025)]] | raw/sources/papers/IL/Zhaxizhuoma - 2025 - FastUMI... | 2026-04-24 |

### Data Efficiency (1)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 17 | [[wiki/sources/data-efficiency/2026-04-23 DemoSpeedup]] | raw/assets/papers/IL/Guo - 2025 - DemoSpeedup...pdf | 2026-04-23 |

### Infrastructure (3)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 1 | [[wiki/sources/infrastructure/2026-04-18 安装Ubuntu双系统]] | raw/sources/blogs/安装Ubuntu双系统.md | 2026-04-18 |
| 2 | [[wiki/sources/infrastructure/2026-04-18 Ubuntu安装实时内核和显卡驱动]] | raw/sources/blogs/Ubuntu安装实时内核和显卡驱动.md | 2026-04-18 |
| 20 | [[wiki/sources/infrastructure/2026-04-24 控制器框架规范v4]] | raw/sources/others/控制器框架规范v4.md | 2026-04-24 |

### Guides & Tools (2)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 12 | [[wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide]] | raw/sources/github/TianxingChenEmbodied-AI-Guide...md | 2026-04-19 |
| 13 | [[wiki/sources/guides-tools/2026-04-19 MinerU Document Explorer]] | raw/sources/github/MinerU-Document-ExplorerREADME-zh.md | 2026-04-19 |

### Lab Automation (5)
| # | Page | Original Source | Date |
|---|------|----------------|------|
| 38 | [[wiki/sources/lab-automation/2026-05-17 ChemBot (Huang 2026)]] | raw/assets/papers/Lab Automation/Huang - 2026 - ChemBot.pdf | 2026-05-17 |
| 39 | [[wiki/sources/lab-automation/2026-05-17 BioMARS (Qiu 2025)]] | raw/assets/papers/Lab Automation/Qiu - 2025 - BioMARS.pdf | 2026-05-17 |
| 40 | [[wiki/sources/lab-automation/2026-05-17 Touch in the Wild (Zhu 2025)]] | raw/assets/papers/Lab Automation/Zhu - 2025 - Touch in the Wild.pdf | 2026-05-17 |
| 41 | [[wiki/sources/lab-automation/2026-05-17 Intelligent Science Laboratory Position (Zhang 2025)]] | raw/assets/papers/Lab Automation/Zhang - 2025 - Intelligent Science Laboratory.pdf | 2026-05-17 |
| 42 | [[wiki/sources/lab-automation/2026-05-17 Scaling Laws Scientific Discovery (Zhang 2025)]] | raw/assets/papers/Lab Automation/Zhang - 2025 - Scaling Laws Scientific Discovery.pdf | 2026-05-17 |

---

## Entities by Category

### Models (3)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/entities/models/π₀.₅]] | Physical Intelligence 第二代 VLA，开放世界泛化能力增强版 | 来源 9, 18 |
| [[wiki/entities/models/SmolVLA]] | HuggingFace 轻量 VLA，基于 SmolVLM 骨干，LeRobot 原生支持，消费级硬件可用 | 来源 9, 18 |
| [[wiki/entities/models/π₀.₇]] | Physical Intelligence 第四代 VLA，多模态 Diverse Prompt Conditioning，开箱媲美 RL 专家，零样本跨机身迁移 | 来源 25 |

### Frameworks (3)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/entities/frameworks/RLinf]] | 开源 RL 训练基础设施，面向具身 AI 与智能体 AI（2.434× 吞吐提升） | 来源 10, 11 |
| [[wiki/entities/frameworks/StarVLA]] | 乐高式模块化 VLA 开发框架（FAST/OFT/PI/GR00T 四变体） | 来源 10, 11 |
| [[wiki/entities/frameworks/ARIS]] | Auto-Research-In-Sleep：轻量 Markdown-only 自主 ML 研究 harness，跨模型对抗评审，65+ skills，覆盖 idea→实验→论文→rebuttal 全流程 | 来源 26 |

### Tools (1)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/entities/tools/Claude Code]] | Anthropic 的 agentic CLI，本知识库和 ARIS harness 的主执行平台，支持 MCP/skills/hooks 生态 | 来源 26 |

### Hardware (4)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/entities/hardware/Ubuntu 20.04]] | 知识库中机器人实验的标准操作系统（20.04.6 LTS） | 来源 1, 2, 3 |
| [[wiki/entities/hardware/Franka Research 3]] | Franka Emika 7-DOF 研究级机械臂（系统版本 5.6.0） | 来源 3, 4 |
| [[wiki/entities/hardware/UMI]] | 手持夹爪 + GoPro 鱼眼 + ORB-SLAM3 的野外数据采集框架，零样本跨机器人迁移（RSS 2024） | 来源 17, 23, 24 |
| [[wiki/entities/hardware/RoboTwin]] | 双臂机器人仿真 & 基准平台（基于 SAPIEN，50 任务自动化数据合成） | 来源 10, 11, 12 |

### Systems (1)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/entities/systems/HIL-SERL]] | UC Berkeley 真实世界 RL 系统，人类在线纠错 + off-policy RL，13 任务 100% 成功率 | 来源 9, 22 |

### People (1)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/entities/people/qq_59001382]] | CSDN 博主，机器人系统配置与算法复现方向 | 来源 1, 2, 3 |

---

## Concepts by Category

### Reinforcement Learning (4)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/concepts/rl/Offline 强化学习]] | 静态数据集下的约束优化 RL 范式 | 来源 6, 7 |
| [[wiki/concepts/rl/RECAP]] | 离线 Advantage-conditioned 策略优化，π₀.₆ 核心方法；补充：条件 I 作用、vs PPO/Actor-Critic、CFGRL 前驱推导 | 来源 7, 10 |
| [[wiki/concepts/rl/DPPO]] | 用 PPO 在线 RL 微调扩散策略的事实基线 | 来源 4, 8 |
| [[wiki/concepts/rl/AWR]] | Advantage-Weighted Regression：KL 正则化 RL 的加权 BC 实现，过滤式模仿学习，Flow Matching 不兼容，RECAP 的直接前驱 | 来源 6, 7 |

### Imitation Learning (2)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/concepts/imitation-learning/HG-DAgger]] | 人类门控的 DAgger 变体，高效采集真实机器人纠错数据 | 来源 7, 10 |
| [[wiki/concepts/imitation-learning/ACT]] | Action Chunking Transformer，基于 Transformer 的动作分块模仿学习策略 | 来源 9, 12, 16 |

### Generative Models (3)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/concepts/generative-models/DDPM]] | 去噪扩散概率模型，现代扩散生成模型的奠基框架（NeurIPS 2020） | 来源 5, 8, 15 |
| [[wiki/concepts/generative-models/Flow Matching]] | 基于 ODE 的生成模型，机器人策略的新默认参数化 | 来源 4, 6, 7 |
| [[wiki/concepts/generative-models/Diffusion Policy]] | 基于扩散模型的机器人模仿学习策略框架（RSS 2023 / IJRR 2024） | 来源 3, 4, 5, 8, 9 |

### VLA (1)
| Page                                            | Summary                                          | Sources             |
| ----------------------------------------------- | ------------------------------------------------ | ------------------- |
| [[wiki/concepts/vla/Vision-Language-Action 模型]] | 视觉+语言+动作统一的机器人基础模型（π 系列、GR00T、LeRobot、StarVLA 等） | 来源 7, 9, 10, 11, 12 |

### Benchmarks (1)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/concepts/benchmarks/LIBERO]] | 机器人操控仿真基准（4 子集：Spatial/Object/Goal/Long），VLA 评测事实标准 | 来源 9, 18, 19 |

### Infrastructure (1)
| Page | Summary | Sources |
|------|---------|---------|
| [[wiki/concepts/infrastructure/PREEMPT_RT实时内核]] | Linux 实时内核补丁，机器人实时控制必备 | 来源 2, 3 |

---

## Comparisons
*Side-by-side analysis of related entities or approaches.*

| Page | Summary |
|------|---------|
| [[wiki/comparisons/RL 微调表达性策略方法对比]] | 9 方法全景（DPPO/ReinFlow/FQL/HIL-SERL/FlowRL/FPO/WarmPrior/OFQL/FAN）：三条 log-prob 解法路线 × On-policy PG vs Offline Q-learning 深度对比 |
| [[wiki/comparisons/VLA RL 微调方法对比]] | 7 种 VLA RL 后训练方法对比：AR-VLA vs Flow-VLA 两路线，RECAP/πRL/VLA-RL/GR-RL/SA-VLA/VLA-OPD/iRe-VLA 四维解法矩阵 |

---

## Analyses
*Deep-dive analyses, answers to complex questions, filed explorations.*

| Page | Summary | Date |
|------|---------|------|
| [[wiki/analyses/Flow Matching 完整数学推导]] | Flow Matching 从 ODE 到 CFM 等价定理的零基础完整数学推导 | 2026-04-21 |
| [[wiki/analyses/DPPO 完整数学推导]] | DPPO 从 MDP/PPO/DDPM 预备知识到双层 MDP、IS 比解析推导、GAE、结构化探索 | 2026-04-23 |
| [[wiki/analyses/DDPM & DDIM 完整数学推导]] | DDPM 前向闭式解、后验配方、ELBO 推导、ε-参数化；DDIM 非马尔可夫过程、边缘一致性 | 2026-04-23 |
| [[wiki/analyses/DemoSpeedup 逐帧熵估计与HDBSCAN聚类数学推导]] | Gaussian KDE 条件动作密度估计、条件熵 Monte Carlo；HDBSCAN 核心距离→聚类稳定性 | 2026-04-24 |
| [[wiki/analyses/ReinFlow π_rl 完整数学推导]] | $\pi_{rl}$ 噪声注入 Markov 化、封闭形式 log-prob 推导、Markov Process PG 定理 4.1 完整证明 | 2026-04-24 |
| [[wiki/analyses/UMI ee6d 位姿变换推理]] | Camera-at-$t_0$ 参考原点下 ee6d 完整 SE(3) 变换链：采集→帧无关性→推理还原→延迟匹配 | 2026-04-25 |
| [[wiki/analyses/π₀.₇ 详细解析]] | VLA / World Model / High-Level Policy 三模型训练目标、推理时 Prompt 组装与 CFG 完整解析 | 2026-04-29 |
| [[wiki/analyses/π₀.₆ 与 RECAP 原理解析]] | π₀.₆ 架构（Gemma 3 4B + 860M flow expert + KI）、RECAP 完整数学推导（贝叶斯反向消除 + delta 二值化 + flow log-likelihood 下界）、三阶段管线与定向失败模式消除 | 2026-05-17 |
| [[wiki/analyses/RL微调生成模型的技术挑战与解决方案]] | BPTT 三重代价 + log-prob 不可解析等根本障碍；三条解法路线（Markov 化 PG / CFM loss ratio / 离线 Q-learning）+ WarmPrior 正交优化轴；含完整数学推导 | 2026-05-18 |

---

## Overview
- [[wiki/overview]] — High-level synthesis of the entire knowledge base.

---

## Knowledge Gaps（待建页）
> 以下概念仅在单一来源中出现，暂不建立独立页面；再次出现时立即创建：

**来自 ARIS（来源 26）**：
- Cross-Model Adversarial Review：跨模型对抗评审，ARIS 核心机制（仅来源 26）
- Harness Engineering：研究流程的系统性工程化（仅来源 26）
- Skills-as-Prompts：将工作流步骤编码为 Markdown prompt（仅来源 26）
- Evidence-to-Claim Audit：多层完整性核查机制（仅来源 26）
- Persistent Research Wiki：跨 session 积累的研究知识库（仅来源 26）
- Meta-Optimization（研究 harness 语境）：分析 agent 自身使用日志并提出改进（仅来源 26）

**来自 iRe-VLA（来源 33）**：
- **iRe-VLA 迭代框架**：RL+SFT 交替迭代的 VLA 后训练范式；仅来源 33

**来自 VLA-RL（来源 34）**：
- **Robotic Process Reward Model（RPRM）**：用 VLM 在任务分段上生成伪过程奖励；仅来源 34
- **Inference Scaling Laws（机器人领域）**：测试时增加优化步骤可持续提升机器人策略性能；仅来源 34

**来自 GR-RL（来源 35）**：
- **Q-value as Task Progress Function**：用离线 RL Q 值过滤次优示范；仅来源 35
- **Morphological Symmetry Augmentation**：利用机械臂几何对称做数据增强；仅来源 35
- **Latent Space Noise Predictor（VLA RL）**：VLA 潜空间残差噪声预测实现高精度 RL；仅来源 35

**来自 SA-VLA（来源 36）**：
- **Spatial Inductive Bias Erosion**：RL 微调流匹配 VLA 时空间归纳偏置被侵蚀的现象；仅来源 36
- **SCAN（Spatially-Conditioned Annealed exploration）**：空间感知退火探索策略；仅来源 36

**来自 VLA-OPD（来源 37）**：
- **On-Policy Distillation（机器人策略后训练）**：在策略自生成轨迹上做教师蒸馏；仅来源 37
- **Entropy Explosion / Entropy Collapse（VLA 训练动态）**：Forward-KL 导致熵爆炸、Hard-CE 导致熵崩溃；仅来源 37

**来自 π₀.₆ RECAP（来源 7）补充**：
- **CFGRL（Frans et al. 2025）**：Classifier-Free Guidance RL，RECAP 的理论前驱；提出用伯努利指示符 $I$ 替代连续权重 $\exp(A/\beta)$ 的"鲜为人知的结果" $(\star\star)$；原始论文尚未入库，仅来源 7 引用。详细说明已整合进 [[wiki/concepts/rl/RECAP]] "CFGRL：RECAP 的理论前驱"节。

**来自 Flow / RL 论文族**：
- ~~**Rectified Flow / Shortcut Models**：仅来源 4（ReinFlow）~~ ✅ **已入 Analysis 页**：[[wiki/analyses/ReinFlow π_rl 完整数学推导]]

**来自 DemoSpeedup（来源 17）**：
- **DemoSpeedup 方法**：仅来源 17；数学推导见 [[wiki/analyses/DemoSpeedup 逐帧熵估计与HDBSCAN聚类数学推导]]
- **Entropy-guided Demonstration Acceleration**：仅来源 17；核心数学已入 Analysis 页
- **HDBSCAN 聚类（机器人策略应用）**：仅来源 17；完整数学推导已入 Analysis 页

**来自 VLASH（来源 18）**：
- **Asynchronous VLA Inference**：仅来源 18
- **Action Quantization（机器人动作粗化）**：仅来源 18
- **RTC（Real-time Chunking）**：仅来源 18 作为比较对象
- **Kinetix 基准**：仅来源 18

**来自 LeRobot（来源 9）**：
- **VQ-BeT**、**Multitask DiT Policy**
- **TDMPC**、**QC-FQL**
- **GR00T N1.5 / XVLA / Pi0Fast**（仅来源 9）
- **LeRobotDataset 格式**、**MetaWorld 基准**
- **SO-100 / Reachy 2 / Unitree G1** 等硬件

**来自 StarVLA（来源 11）**：
- **WM4A（World Model for Action）**：仅来源 11
- **StarVLA-FAST / OFT / PI / GR00T** 各变体详情：仅来源 11

**来自 Embodied-AI-Guide（来源 12）**：
- ~~**RoboTwin 2.0 + SAPIEN**~~ ✅：[[wiki/entities/hardware/RoboTwin]]

**来自 Diffusion-Policy 工程栈（来源 3）**：
- **polymetis**、**libfranka**（下一次有对比出现时建对比页）

**来自 DDPM（来源 15）**：
- ~~**DDIM**~~ ✅：[[wiki/analyses/DDPM & DDIM 完整数学推导]]
- **Score Matching / NCSN**：仅来源 15

**来自 ACT/ALOHA（来源 16）**：
- **ALOHA 硬件平台**（ViperX + WidowX + 3D 打印组件）：仅来源 16
- **MuJoCo 仿真任务**（Transfer Cube、Bimanual Insertion）：仅来源 16

**来自 UMI（来源 23）**：
- **ORB-SLAM3 / Inertial-Monocular SLAM**
- **Inference-time Latency Matching**：来源 23, 24 共同出现，≥2 来源
- **In-the-Wild Robot Data Collection**：来源 23, 24 共同主题

**来自 FastUMI（来源 24）**：
- **Smooth-ACT / PoseACT**：仅来源 24
- **Depth-Enhanced DP（Depth Anything V2 + DP 融合）**：仅来源 24
- **RealSense T265 / RoboBaton MINI**：仅来源 24

**来自 π₀.₇（来源 25）**：
- **MEM（Multi-scale Embodied Memory）**：仅来源 25
- **BAGEL 世界模型**：仅来源 25
- **Episode Metadata Conditioning**：仅来源 25
- **Classifier-Free Guidance for Robot Policies**：仅来源 25
- **Language Coaching（机器人）**：仅来源 25
- ~~**Cross-Embodiment Transfer（跨机身迁移）**~~ ✅：已在 VLA Concept 页简述

**来自 ChemBot（来源 38）**：
- **Skill-VLA**：进度条件化的 VLA 执行器，将任务步骤进度 $p_t$ 作为额外条件输入；仅来源 38
- **Episodic Memory for Robot VLA**：将成功操作轨迹持久化为可检索资产的长期记忆机制；仅来源 38
- **Trajectory Discontinuity（VLA 推理延迟问题）**：VLA 推理延迟导致的动作不连续性；仅来源 38

**来自 BioMARS（来源 39）**：
- **RAG for Robotic Lab Automation**：检索增强生成注入机器人实验规划；仅来源 39
- **Context-Aware Optimization（实验场景）**：基于 VLM 感知动态调整实验参数；仅来源 39

**来自 Touch in the Wild（来源 40）**：
- **Portable Visuo-Tactile Gripper**：集成触觉传感器的便携末端执行器；仅来源 40
- **Cross-Modal Representation Learning（视触觉）**：对比学习对齐视觉与触觉嵌入；仅来源 40
- **In-the-Wild Tactile Data Collection**：野外环境下的触觉演示数据采集范式；仅来源 40

**来自 Intelligent Science Laboratory Position（来源 41）**：
- **Intelligent Science Laboratory (ISL)**：认知 AI + 具身 AI 深度整合的自主科学实验室范式；仅来源 41
- **Cognitive-Embodied Loop（科学发现）**：假设生成→物理实验→结果分析的闭环迭代系统；仅来源 41
- **Experiment Protocol Interface**：LLM 规划器与机器人执行层之间的标准化协议接口；仅来源 41

**来自 Scaling Laws Scientific Discovery（来源 42）**：
- **Autonomous Generalist Scientist**：覆盖科研全流程的自主 AI+机器人系统；仅来源 42
- **Scaling Laws for Scientific Discovery**：AI+机器人结合的科学发现能力规模化法则；仅来源 42
- **Robot Scientist**：具备通用实验室操作能力的机器人科学家；仅来源 42

**⚡ 概念升级提醒（≥2 来源达阈值）**：
- **CFM Loss Ratio 作为 IS Ratio 替代** ✅ 已达 ≥3 来源（#29 McAllister、#43 FPO++、#45 π0-FPO），**待建 Concept 页**：条件流匹配损失比替代重要性采样比，各纸独立提出但核心思想相同。

**来自 FPO++ (Yi 2026，来源 43)**：
- **ASPO（Asymmetric SPO）**：正负 advantage 使用不同 trust region 的非对称 PPO 目标；仅来源 43
- **Per-Sample Flow Ratio**：逐 MC 样本计算 CFM loss ratio 而非先平均；仅来源 43
- **Booster T1**：人形机器人平台，FPO++ sim-to-real 测试对象之一；仅来源 43

**来自 LaST-R1 (Chen 2026，来源 44)**：
- **LAPO（Latent-to-Action Policy Optimization）**：latent embedding + action 联合 PPO 优化，Gaussian IS ratio 近似 latent；仅来源 44
- **Adaptive Latent CoT**：`<latent_end>` 动态发射 + RL 奖励调节推理长度；仅来源 44
- **DINOv3 CLS Latent Targets**：用 DINOv3 CLS token 作为 offline latent 预训练目标；仅来源 44

**来自 π0-FPO (Lyu 2025，来源 45)**：
- **Multi-step Latent (Euler) Exploration**：在 latent 空间做 Euler 扰动产生时序相关探索；仅来源 45
- **Q-Ensemble Critic for Flow-VLA RL**：多 critic 集成在 flow-matching VLA latent 空间的 Q 估计；仅来源 45

**来自 VLAC (Zhai 2025，来源 46)**：
- **VLAC（Vision-Language-Action-Critic）**：actor + critic 统一于单一 VLM（InternVL），token-level 输出 reward delta；仅来源 46
- **Pairwise Progress Delta Reward**：以配对图像时序差值为 dense reward，独立于任务起始点；仅来源 46
- **Graded Human-in-the-Loop Protocol**：三级（offline replay / return&explore / human guide）分级 HIL 干预框架；仅来源 46
