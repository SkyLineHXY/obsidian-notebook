# Wiki Log

> **Chronological, append-only record** of all wiki operations.
> Each entry starts with `## [YYYY-MM-DD] <operation> | <title>` for easy parsing.
>
> Quick commands:
> ```bash
> # Last 5 entries
> grep "^## \[" wiki/log.md | tail -5
> # All ingests
> grep "^## \[.*ingest" wiki/log.md
> # All queries
> grep "^## \[.*query" wiki/log.md
> ```

---

## [2026-04-18] init | Wiki initialized

- Created folder structure based on Karpathy's LLM Wiki methodology.
- Layers: `raw/` (sources + assets), `wiki/` (entities, concepts, sources, comparisons, analyses), `CLAUDE.md` (schema).
- Created `index.md`, `log.md`, `overview.md`.
- Source document: [[karpathy的wiki方法论]]

## [2026-04-18] ingest | 首批 4 篇文章摄取（Ubuntu 系统配置 + Diffusion Policy + ReinFlow）

**新建 Source 页（4 篇）**：
- [[wiki/sources/infrastructure/2026-04-18 安装Ubuntu双系统]] ← raw/sources/blogs/安装Ubuntu双系统.md
- [[wiki/sources/infrastructure/2026-04-18 Ubuntu安装实时内核和显卡驱动]] ← raw/sources/blogs/Ubuntu安装实时内核和显卡驱动.md
- [[wiki/sources/imitation-learning/2026-04-18 复现Diffusion-policy模型]] ← raw/sources/blogs/复现Diffusion-policy模型.md
- [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]] ← raw/sources/papers/ReinFlow...pdf（NeurIPS 2025）

**新建 Entity 页（3 个，均满足 ≥2 来源阈值）**：
- [[wiki/entities/people/qq_59001382]]（来源 1, 2, 3）
- [[wiki/entities/hardware/Ubuntu 20.04]]（来源 1, 2, 3）
- [[wiki/entities/hardware/Franka Research 3]]（来源 3, 4）

**新建 Concept 页（2 个，均满足 ≥2 来源阈值）**：
- [[wiki/concepts/infrastructure/PREEMPT_RT实时内核]]（来源 2, 3）
- [[wiki/concepts/generative-models/Diffusion Policy]]（来源 3, 4）

**暂未建页（单一来源）**：Flow Matching、polymetis、libfranka、Shortcut Models
**更新**：wiki/index.md（Stats、全部表格）

## [2026-04-19] update | CLAUDE.md schema 升级 v1.4 → v1.5

**动机**：raw/ 下出现 `github/` 来源类型、`papers/<topic>/` 二级分类，以及 `wiki/analyses/` 目录，CLAUDE.md 目录图和命名规范未同步。

**修改**：
- §1 目录图修正了此前 `wiki/` 子树被缩进到 `raw/sources/` 下的错误；显式列出 `raw/sources/github/`、`raw/sources/papers/<topic>/`、`wiki/analyses/`、`wiki/overview.md`
- §2.1 Frontmatter `type` 枚举增加 `analysis`
- §2.2 命名表新增 Analysis 行；新增 Source 页命名扩展说明（论文 / 博客 / GitHub 三种）
- 清理文末孤立字符 `x`
- 末尾版本信息更新到 v1.5（保留 v1.4 版本说明用于追溯）

## [2026-04-19] ingest | 第二轮扩展：5 篇 RL / VLA / 框架类来源

**新建 Source 页（5 篇）**：
- [[wiki/sources/imitation-learning/2026-04-19 Diffusion Policy (Chi 2024)]] ← papers/IL(Imitation Learning)/Chi 等 - 2024 - Diffusion Policy.pdf（RSS 2023 / IJRR 2024 原始论文）
- [[wiki/sources/rl-finetuning/2026-04-19 DPPO]] ← papers/VLA+RL/Ren 等 - 2024 - DPPO.pdf
- [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]] ← papers/VLA+RL/Park 等 - 2025 - Flow Q-Learning.pdf（ICML 2025）
- [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] ← papers/VLA+RL/Intelligence 等 - 2025 - π₀.₆.pdf
- [[wiki/sources/frameworks/2026-04-19 LeRobot]] ← github/huggingfacelerobot...md

**新建 Concept 页（4 个，均满足 ≥2 来源阈值）**：
- [[wiki/concepts/generative-models/Flow Matching]]（来源 4, 6, 7；从上一版 Knowledge Gaps 晋升）
- [[wiki/concepts/rl/DPPO]]（来源 4, 8）
- [[wiki/concepts/vla/Vision-Language-Action 模型]]（来源 7, 9）
- [[wiki/concepts/rl/Offline 强化学习]]（来源 6, 7）

**新建 Comparison 页（1 个）**：
- [[RL 微调表达性策略方法对比]]（来源 4, 6, 7, 8 四篇共同主题）

**更新现有 Concept 页**：
- [[wiki/concepts/generative-models/Diffusion Policy]]：sources 从 2 篇 → 5 篇；加入原论文的 DDPM 公式、三大设计决策、RL 微调方法表；交叉链接到所有新建概念页

**维护**：
- wiki/index.md：Stats 更新（9 sources / 6 concepts / 1 comparison），Source 表加编号列，Knowledge Gaps 重写
- wiki/overview.md：重写 Key Themes（加入"两条主流路线"、"四种 RL 微调范式"、"从 polymetis 到 LeRobot"三大主题）；解答上一版 Open Question #1
- wiki/log.md：追加本条

**新识别的 Knowledge Gaps**：
- RECAP / Advantage Conditioning / HG-DAgger（仅 π₀.₆）
- ACT / VQ-BeT / Pi0Fast / Pi0.5 / GR00T N1.5 / SmolVLA / XVLA / HIL-SERL / TDMPC（仅 LeRobot）
- LeRobotDataset / LIBERO / MetaWorld（仅 LeRobot）
- Rectified Flow / Shortcut Models（仍仅 ReinFlow，尚未达标）

## [2026-04-19] ingest | 第三轮扩展：3 篇 GitHub 来源（RLinf / StarVLA / Embodied-AI-Guide）

**新建 Source 页（3 篇）**：
- [[wiki/sources/frameworks/2026-04-19 RLinf]] ← github/RLinf 具身 RL 基础设施 README
- [[wiki/sources/frameworks/2026-04-19 StarVLA]] ← github/starVLA 乐高式 VLA 开发框架 README（arXiv:2604.05014）
- [[wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide]] ← github/TianxingChen 具身智能技术指南

**新建 Entity 页（3 个，均满足 ≥2 来源阈值）**：
- [[wiki/entities/frameworks/RLinf]]（来源 10, 11：RLinf README + StarVLA README 双向引用）
- [[wiki/entities/frameworks/StarVLA]]（来源 10, 11：同上）
-[[RoboTwin]]]（来源 10, 11, 12：RLinf + StarVLA + Embodied-AI-Guide）

**新建 Concept 页（2 个，Knowledge Gaps 晋升）**：
- [[wiki/concepts/rl/RECAP]]（来源 7, 10：π₀.₆ 原始提出 + RLinf 集成支持；含完整数学推导）
- [[wiki/concepts/imitation-learning/HG-DAgger]]（来源 7, 10：π₀.₆ 数据采集策略 + RLinf 真机支持）

**更新现有页**：
- [[wiki/concepts/vla/Vision-Language-Action 模型]]：sources 扩展至 5 篇来源；知识库出现表新增 StarVLA/RLinf/EmbodiedAI-Guide 行；研究问题更新 RECAP/RLinf 链接；删除已解决的 Knowledge Gaps 条目

**维护**：
- wiki/index.md：Stats 更新（12 sources / 6 entities / 8 concepts / 1 comparison），全表增行，Knowledge Gaps 重写
- wiki/log.md：追加本条

**新识别的 Knowledge Gaps**：
- **WM4A**（World Model for Action）：仅来源 11
- **ACT**（Action Chunking Transformer）：LeRobot（9）+ Embodied-AI-Guide（12）→ 已达 ≥2 阈值，可考虑建页

## [2026-04-19] update | CLAUDE.md schema 升级 v1.5 → v1.6

**动机**：用户配置了 `mineru-document-explorer` skill，需要将 PDF/文档解析流程从模糊描述升级为强制性规范，防止 Claude 误用 `Read` 工具直读 PDF。

**修改**：
- §3.1 step 1：重写为明确的 `mineru-document-explorer` skill 强制调用指令，列出禁止条款（直读 PDF、大文档整体 get）
- §5：将原"快速参考 (CLI 速查)"占位节全面替换为《文档解析规范》，包含：Collection 映射表、PDF 标准读取四步流程代码块、决策速查表、Lint 额外检查项
- 版本信息更新至 v1.6，保留历史版本追溯

## [2026-04-19] ingest + update | Wiki 全面更新

**新建 Source 页（2 篇）**：
- [[wiki/sources/guides-tools/2026-04-19 MinerU Document Explorer]] ← github/MinerU-Document-ExplorerREADME-zh.md（文档解析基础设施）
- [[wiki/sources/generative/2026-04-19 Flow Matching (Lipman 2023)]] ← papers/Generative Model/Lipman 等 - 2023 - Flow Matching.pdf（Flow Matching 原始论文）

**新建 Concept 页（1 个，达 ≥2 来源阈值）**：
- [[wiki/concepts/imitation-learning/ACT]]（Action Chunking Transformer）— 来源 9 (LeRobot) + 12 (Embodied-AI-Guide)

**更新现有页**：
- [[wiki/concepts/generative-models/Flow Matching]]：sources 扩展至 4 篇（添加 Lipman 2023），出现表添加原始论文行
- [[wiki/sources/frameworks/2026-04-19 LeRobot]]：ACT 从知识缺口移到链接，表格中添加 ACT wiki-link
- [[wiki/sources/guides-tools/2026-04-19 Embodied-AI-Guide]]：ACT 文本添加 wiki-link
- [[wiki/overview]]：添加"知识库基础设施"主题节，更新 Knowledge Gaps 与 Last Updated
- [[wiki/index.md]]：Stats 更新（14 sources / 9 concepts），全表增行，Knowledge Gaps 标记 ACT 已建页

**维护**：
- wiki/log.md：追加本条

## [2026-04-21] ingest | DDPM (Ho 2020) + ACT/ALOHA (Zhao 2023)

**PDF 解析**（MinerU VLM）：
- Ho 等 - 2020 - Denoising Diffusion Probabilistic Models.pdf → 455 行 Markdown
- Zhao 等 - 2023 - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.pdf → 496 行 Markdown

**新建 Source 页（2 篇）**：
- [[wiki/sources/generative/2026-04-21 DDPM (Ho 2020)]] ← papers/Generative Model/Ho 等 - 2020 - Denoising Diffusion Probabilistic Models.pdf
- [[wiki/sources/imitation-learning/2026-04-21 ACT - ALOHA (Zhao 2023)]] ← papers/IL(Imitation Learning)/Zhao 等 - 2023 - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.pdf

**新建 Concept 页（1 个，达 ≥2 来源阈值）**：
- [[wiki/concepts/generative-models/DDPM]] — 来源 5 (Diffusion Policy Chi 2024) + 8 (DPPO) + 15 (DDPM 原文)；含完整数学推导（§2.6 合规）

**更新现有页**：
- [[wiki/concepts/imitation-learning/ACT]]：frontmatter sources 添加原始论文；References 节更新为正式 wiki-link
- [[wiki/concepts/generative-models/Diffusion Policy]]：正文 DDPM 链接至 [[wiki/concepts/generative-models/DDPM]]；来源出现表新增 DDPM 原文行
- [[wiki/index.md]]：Stats 更新（16 sources / 10 concepts）；Source 表增 2 行；Concepts 表增 DDPM 行；Knowledge Gaps 新增 DDPM 与 ACT/ALOHA 待建条目

**维护**：
- wiki/log.md：追加本条

## [2026-04-23] update | DDPM & DDIM 完整数学推导

**新建 Analysis 页（1 个）**：
- [[wiki/analyses/DDPM & DDIM 完整数学推导]]：零基础完整推导，涵盖：
  - **DDPM 部分**：前向过程逐步加噪定义；高斯线性封闭性推导任意时刻闭式解 $q(x_t|x_0)$；前向后验 $q(x_{t-1}|x_t,x_0)$ 完整配方推导（均值 $\tilde\mu_t$ 与方差 $\tilde\beta_t$）；ELBO 从 Jensen 不等式到三项分解；两个高斯 KL 散度化简 $\mathcal{L}_{t-1}$；ε-预测参数化逐步推导；简化目标 $\mathcal{L}_\text{simple}$ 及其优越性分析；采样算法；与 Score Matching / Langevin 动力学的等价关系。
  - **DDIM 部分**：DDPM 慢采样动机；非马尔可夫前向过程定义；边缘分布一致性严格证明（高斯线性变换法）；采样公式逐步推导；$\sigma_t=0$ 确定性极限与 $\sigma_t=\tilde\beta_t$ 退化为 DDPM 两种特殊情况；连续时间 ODE 极限；子序列跳步加速采样；隐空间球面插值。
  - **综合对比表**：DDPM vs DDIM 全维度（本质/训练/步数/确定性/隐空间/理论基础）；$\eta$ 参数在两者之间平滑插值的公式。

**更新现有页**：
- [[wiki/concepts/generative-models/DDPM]]：新增"完整数学推导"入口章节，链接 Analysis 页
- [[wiki/index.md]]：Analyses 统计 2→3；Analyses 表新增 DDPM & DDIM 行；Knowledge Gaps 中 DDIM 条目标记为已入 Analysis 页

## [2026-04-23] ingest | DemoSpeedup + VLASH 两篇新论文摄取

**MinerU 转换**：
- `raw/assets/papers/IL(Imitation Learning)/Guo 等 - 2025 - DemoSpeedup...pdf` → `raw/sources/papers/IL(Imitation Learning)/Guo 等 - 2025 - DemoSpeedup.../Guo 等...md`（手动补全重命名，转换本身成功）
- `raw/assets/papers/VLA/Tang 等 - 2025 - VLASH...pdf` → `raw/sources/papers/VLA/Tang 等.../Tang 等...md`（顺利完成）

**新建 Source 页（2 个）**：
- [[wiki/sources/data-efficiency/2026-04-23 DemoSpeedup]] ← IL(Imitation Learning) 论文；自监督熵引导示教加速，1.7×–3× 加速同时维持成功率
- [[wiki/sources/vla/2026-04-23 VLASH]] ← VLA 论文（MIT + NVIDIA）；未来状态感知异步推理框架，17.4× 反应延迟提升，2.03× 执行加速

**新建 Entity 页（1 个，满足 ≥2 来源阈值）**：
- [[wiki/entities/models/π₀.₅]]（来源 9 LeRobot + 来源 18 VLASH）：Physical Intelligence 第二代 VLA，开放世界泛化增强版

**更新概念页（3 个）**：
- [[wiki/concepts/imitation-learning/ACT]]：新增 DemoSpeedup 来源，updated → 2026-04-23
- [[wiki/concepts/generative-models/Diffusion Policy]]：新增 DemoSpeedup 来源，updated → 2026-04-23
- [[wiki/concepts/vla/Vision-Language-Action 模型]]：新增 VLASH 来源，updated → 2026-04-23

**更新 index.md**：
- Stats：Sources 16→18，Entities 6→7
- 新增来源 17、18 表格行
- Knowledge Gaps：π₀.₅ 标记已建页；SmolVLA / LIBERO 更新状态；新增 DemoSpeedup / VLASH 来源的 Gap 记录
- 标注遗留 Lint 问题：来源 13（MinerU Document Explorer）wiki 页文件缺失

## [2026-04-21] update | Flow Matching 完整数学推导

**新建 Analysis 页（1 个）**：
- [[wiki/analyses/Flow Matching 完整数学推导]]：零基础完整推导，涵盖 ODE/流映射定义、连续性方程、FM 损失、CFM 技巧、CFM=FM 梯度等价定理完整证明、Rectified Flow 线性路径推导、训练与推理算法、与扩散模型的统一视角

**更新**：
- [[wiki/concepts/generative-models/Flow Matching]]：新增"完整数学推导"入口章节，链接 Analysis 页；更新 updated 日期；知识缺口移除已完成的 Rectified Flow 推导项
- [[wiki/index.md]]：Analyses 统计 0→1；Analyses 表新增一行

## [2026-04-24] update | DemoSpeedup 方法数学推导深度分析

**触发原因**：用户请求将 DemoSpeedup 方法概念更新至 wiki，并对逐帧动作熵估计（Gaussian KDE）与 HDBSCAN 密度聚类分段进行严谨数学梳理。

**新建 Analysis 页（1 个）**：
- [[wiki/analyses/DemoSpeedup 逐帧熵估计与HDBSCAN聚类数学推导]]：完整推导链，涵盖：
  - Gaussian KDE 理论基础（核函数、带宽 $h$、Silverman 法则）
  - 条件动作密度估计（式 1）逐步推导与物理意义
  - 条件动作熵 Monte Carlo 近似（式 2）推导
  - 低熵/高熵极端性质验证
  - ACT（CVAE 隐变量采样）vs DP（DDPM 噪声采样）的采样机制对比
  - HDBSCAN 完整推导：核心距离（式 3）→ 互达距离（式 4）→ MST（式 5）→ Dendrogram → 聚类树压缩 → 稳定性分数 FOSC（式 6-7）→ GLOSH 离群点分数（式 8）
  - HDBSCAN vs K-means/DBSCAN/GMM 的选型分析
  - 两个组件协作逻辑全链路图

**更新现有页（2 个）**：
- [[wiki/sources/data-efficiency/2026-04-23 DemoSpeedup]]：tags 补充 KDE/HDBSCAN；updated → 2026-04-24；新增"深度分析"章节，链接 Analysis 页
- [[wiki/index.md]]：Analyses 统计 3→4；Analyses 表新增一行；Knowledge Gaps 中 DemoSpeedup 相关条目注明已入 Analysis 页

## [2026-04-24] update | ReinFlow $\pi_{rl}$ 完整数学推导深度分析

**触发原因**：用户请求详细分析 $\pi_{rl}$ 以及 ReinFlow 的数学原理，并保存到 wiki。

**新建 Analysis 页（1 个）**：
- [[wiki/analyses/ReinFlow π_rl 完整数学推导]]：完整推导链，涵盖：
  - §0 符号定义表（14 个核心符号）
  - §1 问题背景：POMDP 框架；Rectified Flow / Shortcut Models 数学形式；两大技术障碍（log-prob 不可解析、无探索）
  - §2 $\pi_{rl}$ 的构造：噪声注入机制（Eq.6）→ Markov 化 → 联合分布 → $\pi_{rl}$ 边缘定义；封闭形式 log 概率（Eq.7）完整展开
  - §3 Markov Process PG 定理 4.1 三阶段完整证明：
    - 阶段一：标准 PG（POMDP 对数导数技巧 + 求和顺序交换 → Eq.14）
    - 阶段二：扩展至 Markov Process 参数化（Eq.16 边缘概率展开 → Eq.17）
    - 阶段三：平稳 POMDP 重要性引理（Eq.19 折扣访问频率 → Eq.21 最终形式）
  - §4 PPO 实现：Clipped Surrogate Loss + IS 比封闭计算
  - §5 正则化：W2 距离上界（Eq.12）；熵正则化（负块熵率 + 高斯微分熵封闭形式）
  - §6 训练-推理非对称性：噪声网络生命周期图
  - §7 与 DPPO 的数学对比表

**更新现有页（2 个）**：
- [[wiki/sources/rl-finetuning/2026-04-18 ReinFlow]]：tags 补充 Markov Process/噪声注入；updated → 2026-04-24；新增"深度分析"章节，链接 Analysis 页；完善关联知识链接
- [[wiki/index.md]]：Analyses 统计 4→5；Analyses 表新增一行；Knowledge Gaps 中 Rectified Flow/Shortcut Models 标记已入 Analysis 页

---

## [2026-04-24] ingest | 批量更新：RLinf文档、控制器框架规范v4；新建 SmolVLA、LIBERO

**触发原因**：用户请求更新 wiki，经扫描发现 2 个未摄取新来源 + 2 个达到阈值的 Knowledge Gap 条目。

**新建 Source 页（2 个）**：
- [[wiki/sources/frameworks/2026-04-24 RLinf文档]]：RLinf 官方文档（ReadTheDocs 中文版）。提炼 M2Flow 范式（宏观逻辑/微观调度解耦）、三种执行模式（共享/分离/混合）、自动调度机制、全量支持的 RL 算法（PPO/GRPO/SAC 等 9 种）、支持的 VLA 模型（OpenVLA/π₀/GR00T-N1.5）、内置仿真环境（LIBERO/ManiSkill3/IsaacLab）、双后端架构（FSDP+HF / Megatron+SGLang）、高级特性（5D 并行/LoRA/异构集群）、关联论文体系（含未摄取的 πRL 论文）。
- [[wiki/sources/infrastructure/2026-04-24 控制器框架规范v4]]：mindrealm 中台控制器框架工程规范。提炼 4 层架构图、5 组强制接口方法、全局池 + 工厂函数三件套模式、4 类设备决策树（TCP 长连接/SDK 包装/Modbus 多从站/串口/HTTP）、VirtualInstanceController 代理模式、各类型核心注意事项。

**新建 Entity 页（1 个）**：
- [[wiki/entities/models/SmolVLA]]：HuggingFace 轻量 VLA。触发条件：来源 9（LeRobot）+ 来源 18（VLASH）≥2 阈值。内容覆盖：Smol 模型家族定位、与大型 VLA 的参数/硬件/成本对比、在 LeRobot 和 VLASH 中的出现上下文、知识缺口说明。

**新建 Concept 页（1 个）**：
- [[wiki/concepts/benchmarks/LIBERO]]：机器人操控仿真基准（NeurIPS 2023）。触发条件：来源 9+18+19 共 3 次出现。内容覆盖：LIBERO 4 子集定义（Spatial/Object/Goal/Long）、在 LeRobot/VLASH/RLinf 中的具体使用场景、与 π₀.₅ VLASH 实验的性能数据（96.8%→97.1% 同步→异步）、与 MetaWorld/ManiSkill/RoboTwin 的基准对比表。

**更新现有页（2 个）**：
- [[wiki/entities/frameworks/RLinf]]：新增 `wiki/sources/frameworks/2026-04-24 RLinf文档` 和 `wiki/concepts/benchmarks/LIBERO` 的关联链接
- [[wiki/index.md]]：Stats 更新（Sources 18→20, Entities 7→8, Concepts 10→11）；新增 2 条 Source 行、1 条 Entity 行、1 条 Concept 行；Knowledge Gaps 中 SmolVLA 和 LIBERO 基准标记已建页

**待摄取（下次 Ingest）**：
- `raw/assets/papers/VLA+RL/Chen 等 - 2026 - πRL Online RL Fine-tuning for Flow-based VLAs.pdf`（需先运行 mineru skill 转换）

## [2026-04-24] ingest | πRL + HIL-SERL 两篇新论文

- 识别出 2 篇已完成 MinerU 转换但尚未建立 wiki 页面的新论文
- **来源 21** [[wiki/sources/vla-rl/2026-04-24 πRL]] 创建：基于 `raw/sources/papers/VLA+RL/Chen 等 - 2026 - πRL...md`
  - 内容覆盖：Flow-Noise / Flow-SDE 两条路线、双层 MDP 数学框架、LIBERO/ManiSkill/MetaWorld/CALVIN 实验结果
- **来源 22** [[wiki/sources/rl-finetuning/2026-04-24 HIL-SERL]] 创建：基于 `raw/sources/papers/VLA+RL/Luo 等 - 2025 - HIL-SERL...md`
  - 内容覆盖：RLPD 算法、人类在线纠错机制、13 任务 100% 成功率、Q-value 漏斗机制分析
- **新建实体页** [[wiki/entities/systems/HIL-SERL]]：LeRobot (来源 9) + Luo 等 2025 (来源 22) 满足 ≥2 阈值
- **更新概念页** [[wiki/concepts/imitation-learning/HG-DAgger]]：新增 HIL-SERL 与 HG-DAgger 对比段落（RL vs SL 利用纠错数据的本质差异）
- [[wiki/index.md]] 更新：Stats Sources 20→22, Entities 8→9；新增 2 条 Source 行、1 条 Entity 行；Knowledge Gaps 中 HIL-SERL 标记已建页

## [2026-04-24] ingest | UMI + FastUMI 两篇 IL 论文

**MinerU 转换**（本次 session 前已完成）：
- `raw/assets/papers/IL(Imitation Learning)/Chi 等 - 2024 - Universal Manipulation Interface...pdf` → `raw/sources/papers/IL(Imitation Learning)/Chi 等...md`（路径长度超 MAX_PATH，手动修复重命名）
- `raw/assets/papers/IL(Imitation Learning)/Zhaxizhuoma 等 - 2025 - FastUMI...pdf` → `raw/sources/papers/IL(Imitation Learning)/Zhaxizhuoma 等...md`（同上）

**新建 Source 页（2 个）**：
- **来源 23** [[wiki/sources/data-collection/2026-04-24 UMI (Chi 2024)]]：手持夹爪 + GoPro 鱼眼 + ORB-SLAM3 + Diffusion Policy 野外数据采集框架；RSS 2024；$371 成本；Cup Arrangement 100%、Dynamic Tossing 87.5%、野外泛化 71.7%
- **来源 24** [[wiki/sources/data-collection/2026-04-24 FastUMI (Zhaxizhuoma 2025)]]：UMI 的全面工程重设计；T265 替换 VIO；10,000 条 / 22 任务开源数据集；Smooth-ACT / PoseACT / Depth-Enhanced DP 三种算法适配

**新建 Entity 页（1 个，满足 ≥2 来源阈值）**：
- [[wiki/entities/hardware/UMI]]：来源 17（DemoSpeedup 使用 UMI 数据）+ 来源 23（原论文）+ 来源 24（FastUMI）三次出现

**更新概念页（2 个）**：
- [[wiki/concepts/generative-models/Diffusion Policy]]：sources 新增来源 23、24；updated → 2026-04-24
- [[wiki/concepts/imitation-learning/ACT]]：sources 新增来源 23（UMI 提及可替代）、24（FastUMI Smooth-ACT / PoseACT）；updated → 2026-04-24

**更新 index.md**：
- Stats：Sources 22→24，Entities 9→10
- 新增来源 23、24 表格行；新增 UMI 实体行
- Knowledge Gaps：新增 UMI / FastUMI 相关 Gap 记录；UMI 实体标记已建页

## [2026-04-25] query | UMI ee6d 位姿变换推理分析

**问题**：UMI 采集的 ee6d 以相机 $t_0$ 位姿为参考原点，推理时如何变换才能让机械臂完成演示任务？

**新建 Analysis 页（1 篇）**：
- [[wiki/analyses/UMI ee6d 位姿变换推理]] ← 用户问询触发

**分析要点**：
- 定义四坐标系 $\{W\}, \{C\}, \{E\}, \{B\}$，明确 SLAM 世界系仅存在于采集阶段
- 推导 Camera-at-$t_0$ 参考的核心公式：$\delta_t = {}^W T_C(t_0)^{-1} \cdot {}^W T_E(t) = {}^{C(t_0)} T_{E(t)}$
- 证明相对表示的帧无关性（Step 2），即绝对坐标系在变换链中消除
- 给出推理核心等式：${}^B T_E(t_k) = {}^B T_E(t_0) \cdot ({}^C T_E)^{-1} \cdot \delta_k$
- 对比 Camera-at-$t_0$ 与 EE-at-$t_0$ 两种参考的等价关系及各自适用场景
- 标记 6 个关键陷阱：外参一致性、重力方向、延迟匹配、初始 ee6d 非零、SLAM 漂移

**更新 index.md**：
- Stats：Analyses 5→6
- 新增 UMI ee6d 位姿变换推理分析行

## [2026-04-25] update | UMI ee6d 位姿变换推理 — 新增 §9 Franka 具体实现

**触发原因**：用户追问"Franka 机械臂如何读取 ${}^B T_F$，以及 E = 法兰时如何处理"。

**更新 Analysis 页（1 个）**：
- [[wiki/analyses/UMI ee6d 位姿变换推理]]：新增 §9「Franka 具体实现：如何读取 ${}^B T_E$」，涵盖：
  - §9.1 Franka 内部坐标系五级链（$\{O\} \to \{F\} \to \{NE\} \to \{EE\}$）及各帧对应的 libfranka 字段
  - §9.2 法兰帧（`panda_link8`）轴方向说明（+Z 指向法兰外，即工具接近方向）
  - §9.3 读取 ${}^B T_F$ 的三种实现方法：libfranka C++（`model.pose(kFlange, state)`）、polymetis Python（`get_ee_pose()`）、ROS tf2（`lookup_transform`）
  - §9.4 ⚠️ 列主序（Column-Major）陷阱及正确 `reshape` 示例
  - §9.5 E = 法兰时的推理核心公式：${}^B T_F(t_k) = {}^B T_F(t_0) \cdot ({}^C T_F)^{-1} \cdot \delta_k$
  - §9.6 ⚠️ 手眼标定对齐规则：标定"手"参考点必须与 ee6d 的 $\{E\}$ 定义一致（法兰 vs TCP）
- frontmatter tags 新增 `Franka`；updated 字段注明修改内容

## [2026-04-29] update | π₀.₇ 训练逻辑全景解析（Analysis 页）

**触发原因**：用户就 π₀.₇ 整体训练逻辑提问，要求将分析存入 wiki/analyses。

**新建 Analysis 页（1 个）**：
- [[π₀.₇ 详细解析]]：完整解析三模型协作机制，涵盖：
  - §1 核心结论：三个模型独立训练，推理时通过 Prompt 组装协同（附 Algorithm 1 时序图）
  - §2 VLA 训练逻辑：KI 双损失分离（VLM 骨干 CE + Action Expert CFM），梯度边界说明，CFM 线性插值路径与向量场的完整数学推导
  - §3 多模态 Prompt $\mathcal{C}_t$ 的四组件构成与精确 Dropout 比例（含 train-test 分布对齐设计）
  - §4 World Model 独立 CFM 训练目标（Eq.2），BAGEL 双路径架构（ViT + VAE），推理异步调用与 1.25 秒延迟优化
  - §5 High-Level Policy 训练：CE 损失 + 两类数据来源（轨迹标注 + Language Coaching 蒸馏），与 VLA 的语义/运动层解耦
  - §6 推理时 Classifier-Free Guidance（CFG）完整公式（Eq.3），β 值选择，attention tree 并行实现技巧
  - §7 三模型训练目标汇总对比表
  - §8 训练与推理阶段"结合"方式详解（数据构建层 + Prompt 组装层）
  - §9 设计选择的直觉解释表
  - §10 完整数学符号定义表（18 个符号）

**更新 index.md**：
- Stats：Analyses 6→7
- 新增 π₀.₇ 训练逻辑全景解析行

---

## [2026-04-29] ingest | π₀.₇ a Steerable Generalist Robotic Foundation Model with Emergent Capabilities

**触发原因**：用户请求更新 wiki，扫描 raw/assets/papers/ 发现 1 篇未摄取新论文。

**MinerU 转换**：
- 源文件：`raw/assets/papers/VLA/Intelligence 等 - 2026 - π₀.₇ a Steerable Generalist Robotic Foundation Model with Emergent Capabilities.pdf`
- 输出目录：`raw/sources/papers/VLA/`（首次尝试因 SOCKS 代理失败，禁用代理后重试成功，耗时 4.2 分钟）
- 产物：`raw/sources/papers/VLA/Intelligence 等 - 2026 - $π_{0.7}$.../$π_{0.7}$....md`

**新建 Source 页（1 个）**：
- [[wiki/sources/vla/2026-04-29 π₀.₇]]（来源 #25）：Physical Intelligence 第四代 VLA。内容覆盖：核心问题与动机（泛化模型 vs 专家模型的性能差距）；多模态 Diverse Prompt Conditioning 四组件（subtask instructions / subgoal images / episode metadata / control mode）；World Model（BAGEL 初始化）subgoal 生成训练细节；混合质量数据训练配方；Classifier-Free Guidance 推理算法（Algorithm 1）；开箱即用灵巧操作 / 指令跟随 / 零样本跨机身迁移（匹配人类专家）/ 组合任务泛化 / language coaching 五大实验结果；数据多样性 scaling 分析。

**新建 Entity 页（1 个）**：
- [[wiki/entities/models/π₀.₇]]：Physical Intelligence 第四代 VLA 实体页，π 系列演进对比，五类涌现能力概述

**新增 Knowledge Gaps（6 条，来自来源 25）**：
- MEM（Multi-scale Embodied Memory）架构
- BAGEL 世界模型
- Episode Metadata Conditioning 训练范式
- Classifier-Free Guidance for Robot Policies
- Language Coaching（机器人）
- Cross-Embodiment Transfer（跨机身迁移）—— 待评估是否建 Concept 页

**更新 index.md**：
- Stats：Sources 24→25，Entities 10→11
- 新增来源 25 表格行；新增 π₀.₇ 实体行；Knowledge Gaps 补充 6 条来自来源 25 的新概念

---

## [2026-05-06] update | UMI ee6d — 重构为纯 EE（法兰）-at-t0 方案

- 删除 Camera-at-$t_0$ 相关内容（原 §3.2、§3.3、§4、§5、§6.4、§8 旧版、§9 旧版、§10）
- 保留并重构为单一方案：EE-at-$t_0$，$\Delta_k = {}^{F(t_0)} T_{F(t_k)}$
- 核心等式：${}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k$，推理时无需外参
- 新增 §9.4 数据预处理代码（从 SLAM 直接输出 EE-at-t0 格式）
- §9.6 推理骨架更新（移除外参计算步骤）

---

## [2026-05-17] update | Wiki taxonomy refactor + ARIS ingest

- 重组 wiki/sources/、wiki/entities/、wiki/concepts/ 进入主题子目录（详见 CLAUDE.md §1 新版目录树）
- 移动 25 sources + 11 entities + 11 concepts 进入 22 个新建子目录，修复 417 处路径式 wikilink
- 新建子目录结构：
  - sources: agent-systems / rl-finetuning / generative / imitation-learning / vla / frameworks / data-collection / data-efficiency / infrastructure / guides-tools
  - entities: models / frameworks / tools / hardware / systems / people
  - concepts: rl / imitation-learning / generative-models / vla / benchmarks / infrastructure
- Ingest: 2026-05-16 ARIS（GitHub README + arXiv:2605.03042，HuggingFace Daily Paper #1）
- 新建 entity pages: [[wiki/entities/frameworks/ARIS]]（harness 本身）, [[wiki/entities/tools/Claude Code]]（主 executor 平台，首次以独立实体出现）
- ARIS source 页覆盖：TL;DR、三层架构、跨模型评审机制、5 Workflows、Evidence-to-Claim 审计级联、65+ skills 全景、5 条设计原则、生态与可移植性、个人观察
- Concept pages 延迟创建：ARIS 相关 6 个新概念（cross-model adversarial review 等）当前仅 1 个 source，待 ≥2 来源触发
- 同步更新 CLAUDE.md §1（目录树）、§2.2（存储路径表 + "根据主题选子目录"规则）、§4.1（索引示例）、Schema 版本 v1.8→v1.9
- wiki/index.md 全面重写：按主题分组的节段式 Source 表（10 个主题 + 新增 Agent Systems 节）、按类别分组的 Entity 表（6 个类别）、按类别分组的 Concept 表（6 个类别）、新增 ARIS Knowledge Gaps 条目

---

## [2026-05-17] ingest | BayesianVLA (Lian 2026) — agent 自动检索演示

**触发原因**：演示 ARIS × Obsidian 集成工作流。通过 HuggingFace Daily Papers MCP 自动检索 VLA 最新论文，agent 直接写入 wiki，无需用户手动查找。

**检索来源**：HuggingFace paper_search MCP（query: "vision language action model robot manipulation"）

**新建 Source 页（1 个）**：
- [[wiki/sources/vla/2026-05-17 BayesianVLA (Lian 2026)]]（来源 #27）：贝叶斯分解 VLA，识别 Information Collapse 问题，引入 Latent Action Queries + Conditional PMI 目标，OOD SimplerEnv +11.3%。

**新增 Knowledge Gaps（3 条）**：
- **Information Collapse**：VLA 训练数据集偏差导致语言-动作互信息趋零；仅本来源系统命名
- **Latent Action Queries**：用于视觉先验/语言后验双分支解耦的可学习 token；仅本来源
- **Conditional PMI for Robot Policy**：将 PMI 最大化用作 VLA 语言基础训练目标；仅本来源

**更新 index.md**：Stats Sources 26→27；VLA 节新增第 27 行

---

## [2026-05-17] ingest | 5 篇 RL+流策略论文批量摄取（agent 自动检索）

**触发原因**：用户请求检索 RL 微调生成模型（ReinFlow / FQL 等方向）的最新文章并下载原文；agent 通过 HuggingFace paper_search + arXiv 下载完成。

**检索来源**：HuggingFace paper_search MCP + arXiv 直接下载（tools/arxiv_fetch.py，已修复代理绕过问题）

**PDF 下载目录**：`raw/assets/papers/VLA+RL/`

**新建 Source 页（5 个）**：
- [[wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026)]]（来源 #28）：RL + 扩散/流策略统一 taxonomy + JAX 模块化基础设施，跨 Gym-Locomotion / DMControl / IsaacLab 标准化基准。
- [[wiki/sources/rl-finetuning/2026-05-17 FPO Flow Matching Policy Gradients (McAllister 2025)]]（来源 #29）：以 CFM loss ratio 替代 IS ratio 做 PPO-clip，无需 log-prob，与采样器解耦。
- [[wiki/sources/rl-finetuning/2026-05-17 WarmPrior (Kang 2026)]]（来源 #30）：以近期动作历史替换 Gaussian 源分布，路径更直，BC 和 prior-space RL 均受益。
- [[wiki/sources/rl-finetuning/2026-05-17 OFQL (Nguyen 2025)]]（来源 #31）：将 DQL 重表述为 Flow Matching，学习 average velocity field 实现 one-step 离线 RL，无需蒸馏。
- [[wiki/sources/rl-finetuning/2026-05-17 FAN (Lee 2026)]]（来源 #32）：单步流策略 + 单噪声样本分布式 critic，offline RL SOTA，含理论收敛保证。

**新增 Knowledge Gaps（6 条）**：
- **FlowRL Taxonomy**：RL + 扩散/流策略方法族的统一分类框架（来源 28）
- **CFM Loss Ratio 作为 IS Ratio 替代**：条件流匹配损失比替代重要性采样比（来源 29）
- **WarmPrior（时序动作先验）**：以近期动作历史为流匹配源分布（来源 30）
- **Prior-Space RL**：在流匹配先验空间定义探索分布的 RL 范式（来源 30）
- **Marginal Average Velocity Field**：对条件速度场取边缘期望实现单步生成（来源 31）
- **Noise-Conditioned Critic**：以随机噪声为 critic 输入实现单样本分布式价值估计（来源 32）

**更新 index.md**：Stats Sources 27→32；RL Fine-Tuning 节 6→11 行

## [2026-05-17] ingest | RL 微调 VLA 系列论文 — 用户主动检索，新增 5 篇

**触发原因**: 用户请求检索近期 RL 方法微调 VLA 的文章并更新 wiki
**检索来源**: HuggingFace paper_search MCP（3 次并行查询）+ arXiv 下载
**PDF 下载目录**: `raw/assets/papers/VLA+RL/`（5 篇全部下载成功）

**新建 Source 页（5 个）**:
- [[wiki/sources/vla-rl/2026-05-17 iRe-VLA (Guo 2025)]]（来源 #33）：iRe-VLA，RL+SFT 交替迭代稳定 VLA 在线 RL 微调的早期探索
- [[wiki/sources/vla-rl/2026-05-17 VLA-RL (Lu 2025)]]（来源 #34）：VLA-RL，轨迹级 RL + VLM 过程奖励模型，OpenVLA-7B 超越 SFT baseline +4.5%
- [[wiki/sources/vla-rl/2026-05-17 GR-RL (Li 2025)]]（来源 #35）：GR-RL，三阶段（离线 RL 过滤+形态对称增强+在线 RL 精调），穿鞋带任务 83.3% 成功率
- [[wiki/sources/vla-rl/2026-05-17 SA-VLA (Pan 2026)]]（来源 #36）：SA-VLA，解决流匹配 VLA 在 RL 微调中空间归纳偏置侵蚀问题
- [[wiki/sources/vla-rl/2026-05-17 VLA-OPD (Zhong 2026)]]（来源 #37）：VLA-OPD，Reverse-KL on-policy 蒸馏桥接 SFT 与在线 RL

**新增 Knowledge Gaps（11 条）**: iRe-VLA 框架、RPRM、Inference Scaling Laws（robot）、Q-value Task Progress、Morphological Symmetry Aug、Latent Noise Predictor、Spatial Inductive Bias Erosion、SCAN、On-Policy Distillation（robot）、Entropy Explosion/Collapse（各仅单一来源）

**更新 index.md**: Stats Sources 32→37；RL Fine-Tuning 节 11→16 行

## [2026-05-17] update | 知识库结构重组：RL Fine-Tuning 拆分为 VLA+RL / 生成模型+RL

**触发原因**: 用户要求将 VLA+RL 与传统生成模型+RL 分开管理

**新建目录**: `wiki/sources/vla-rl/`

**迁移文件（7 篇，从 rl-finetuning/ 移至 vla-rl/）**:
- [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]] — offline advantage RL 应用于 π₀.₆ VLA
- [[wiki/sources/vla-rl/2026-04-24 πRL]] — π 系列 VLA 的在线 RL 微调
- [[wiki/sources/vla-rl/2026-05-17 iRe-VLA (Guo 2025)]] — RL+SFT 交替迭代 VLA
- [[wiki/sources/vla-rl/2026-05-17 VLA-RL (Lu 2025)]] — VLM 过程奖励 + VLA RL
- [[wiki/sources/vla-rl/2026-05-17 GR-RL (Li 2025)]] — GR00T VLA 三阶段 RL
- [[wiki/sources/vla-rl/2026-05-17 SA-VLA (Pan 2026)]] — 空间偏置侵蚀 Flow VLA RL
- [[wiki/sources/vla-rl/2026-05-17 VLA-OPD (Zhong 2026)]] — Reverse-KL 蒸馏 VLA

**留守 rl-finetuning/（9 篇，扩散/Flow 策略 + 传统 SAC 基线）**: ReinFlow、FQL、DPPO、HIL-SERL、FlowRL、FPO、WarmPrior、OFQL、FAN

**交叉引用修复**: 批量替换 20 个文件中的路径引用（`rl-finetuning/` → `vla-rl/`，仅针对 7 篇迁移论文）

**分类依据**:
- **VLA+RL**：策略骨干为 LLM/VLM，RL 目标是大模型级别的策略后训练
- **生成模型+RL**：策略骨干为独立的扩散/Flow 网络（无 LLM），RL 解决连续去噪链的优化问题

## [2026-05-17] ingest | Lab Automation + Robot Learning — 5 篇论文

## [2026-05-17] ingest | Lab Automation + Robot Learning — 5 篇论文

**触发原因**: 用户检索"实验室自动化结合机器人学习方法（模仿学习、VLA、RL 微调）"相关近期论文
**检索来源**: HuggingFace paper_search MCP（多轮检索）
**PDF 下载目录**: 无（仅 HuggingFace 链接，未下载 PDF）

**新建 Source 页（5 个）**:
- [[wiki/sources/lab-automation/2026-05-17 ChemBot (Huang 2026)]]（来源 #38）：VLA + 双层记忆架构用于化学实验室长时域自动化
- [[wiki/sources/lab-automation/2026-05-17 BioMARS (Qiu 2025)]]（来源 #39）：LLM+VLM 层级多智能体系统用于自主生物实验
- [[wiki/sources/lab-automation/2026-05-17 Touch in the Wild (Zhu 2025)]]（来源 #40）：视触觉模仿学习，涵盖移液管等实验室精密操作
- [[wiki/sources/lab-automation/2026-05-17 Intelligent Science Laboratory Position (Zhang 2025)]]（来源 #41）：认知 AI + 具身 AI 整合的 ISL 技术路线图 position paper
- [[wiki/sources/lab-automation/2026-05-17 Scaling Laws Scientific Discovery (Zhang 2025)]]（来源 #42）：AI+机器人科学家的规模化法则

**新增 Knowledge Gaps（14 条）**:
- Skill-VLA、Episodic Memory for Robot VLA、Trajectory Discontinuity（来源 38）
- RAG for Robotic Lab Automation、Context-Aware Optimization（来源 39）
- Portable Visuo-Tactile Gripper、Cross-Modal Representation Learning、In-the-Wild Tactile Data Collection（来源 40）
- Intelligent Science Laboratory (ISL)、Cognitive-Embodied Loop、Experiment Protocol Interface（来源 41）
- Autonomous Generalist Scientist、Scaling Laws for Scientific Discovery、Robot Scientist（来源 42）

**更新 index.md**: Stats Sources 37→42；新增 Lab Automation 节（#38-42）；新增 Knowledge Gaps 块

---

## [2026-05-17] query | RL 微调 VLA + 生成模型全景对比分析

**触发原因**：用户请求详细分析对比 wiki 中 RL 微调 VLA 以及 RL 微调生成模型的所有方法，并更新对比页。

**全量阅读来源**（16 篇）：
- VLA+RL 路线（7 篇）：RECAP/π₀.₆、πRL、iRe-VLA、VLA-RL、GR-RL、SA-VLA、VLA-OPD
- 生成模型+RL 路线（9 篇）：DPPO、ReinFlow、FQL、HIL-SERL、FlowRL、FPO、WarmPrior、OFQL、FAN

**更新 Comparison 页（1 个）**：
- [[wiki/comparisons/RL 微调表达性策略方法对比]]：从 4 方法扩展到 9 方法，新增 FlowRL taxonomy 视角、三条 log-prob 解法路线（Markov 化 / CFM ratio / Q-learning 回避）、On-policy PG vs Offline Q-learning 深度对比、WarmPrior 正交轴说明，以及完整的定量对照表和更新版选型建议

**新建 Comparison 页（1 个）**：
- [[wiki/comparisons/VLA RL 微调方法对比]]：7 种 VLA RL 后训练方法的首次系统对比，覆盖：
  - AR-VLA vs Flow-VLA 两条架构路线的结构性差异
  - 三大技术挑战（log-prob、稀疏奖励、灾难性遗忘）的解法矩阵
  - 四大设计哲学分歧（PG vs Conditioning / 在线 vs 离线 / 奖励来源 / 通用 vs 专精）
  - VLA-OPD 对 KL 目标的深度分析（熵爆炸 vs 熵崩溃 vs Reverse-KL 有界模式搜索）
  - 7 方法演化谱系图 + 完整定量对比 + 分场景选型建议

**更新 index.md**：Stats Comparisons 1→2；Comparisons 表新增 VLA RL 微调方法对比行；既有对比行更新摘要描述

## [2026-05-17] ingest | Lab Automation PDF 下载 + MinerU 转换

**触发原因**: 用户要求获取 5 篇实验室自动化论文的 PDF 并转为 Markdown
**检索来源**: arXiv 直接下载（arxiv_fetch.py，bypass SOCKS 代理）
**MinerU 模式**: vlm（5 并发），耗时 1.6 分钟

**PDF 下载目录**: 
aw/assets/papers/Lab Automation/（新建目录）
- Huang - 2026 - ChemBot.pdf（2921 KB）
- Qiu - 2025 - BioMARS.pdf（14214 KB）
- Zhu - 2025 - Touch in the Wild.pdf（10874 KB）
- Zhang - 2025 - Intelligent Science Laboratory.pdf（1531 KB）
- Zhang - 2025 - Scaling Laws Scientific Discovery.pdf（9292 KB）

**Markdown 输出目录**: 
aw/sources/papers/Lab Automation/（新建目录）
- Huang - 2026 - ChemBot/Huang - 2026 - ChemBot.md
- Qiu - 2025 - BioMARS/Qiu - 2025 - BioMARS.md
- Zhu - 2025 - Touch in the Wild/Zhu - 2025 - Touch in the Wild.md
- Zhang - 2025 - Intelligent Science Laboratory/Zhang - 2025 - Intelligent Science Laboratory.md
- Zhang - 2025 - Scaling Laws Scientific Discovery/Zhang - 2025 - Scaling Laws Scientific Discovery.md

**更新 wiki source 页（5 个）**: sources 字段 HF 链接 → PDF 路径，补充 raw 文件 wiki-link
**更新 index.md**: Lab Automation 节 Original Source 列更新为本地 PDF 路径


## [2026-05-17] query | RECAP 强化学习训练原理详解
- 读取 [[wiki/concepts/rl/RECAP]] 与 [[wiki/sources/vla-rl/2026-04-19 π0.6 RECAP]]，综合给出 RECAP 训练原理的详细分析
- 覆盖：贝叶斯后验视角 → AWR 等价 → 二值化简化 → Prefix token 注入 → 三阶段管线 → 与 DPPO/FQL/ReinFlow 对比
- 未新建页面（既有 Concept 页已涵盖核心内容）


## [2026-05-17] update | 新建 Analysis 页：π₀.₆ 与 RECAP 训练原理全景解析
- 基于原论文 (arXiv:2511.14759, 全文 + Appendix C/D/E/F) 创建独立分析页
- 覆盖：π₀.₅→π₀.₆→π₀.₆* 架构升级、RECAP 严谨数学推导（贝叶斯反向消除 + Delta 二值化 + flow matching 对数似然下界 Eq.4/9）、价值函数设计（B=201 分布式 + 670M backbone + MC-CE loss）、三阶段管线（Algorithm 1）、推理时 β=1 直采 vs β>1 CFG、RECAP vs AWR vs PPO 工程差异、12 节深度解析
- 强化数学严谨性：CFGRL 那条"鲜为人知的结果"完整推导、advantage 阈值 ε_ℓ 的设定规则、N-step vs MC advantage 在 pre-train/post-train 的切换
- 更新 index.md: Analyses 节 +1 行，Stats 7→8


## [2026-05-17] update | 新建 AWR Concept 页；更新 RECAP Concept 页（Q&A 沉淀）
- **新建** [[wiki/concepts/rl/AWR]]：AWR 完整概念页，含 KL 正则化 RL 闭式推导（Step 1-4）、过滤式模仿学习本质、局限性对比表、π₀.₆ 实验数据
  - 阈值检查：AWR 出现于来源 6（Flow Q-Learning）+ 来源 7（π₀.₆），满足 ≥2 来源建页条件
- **更新** [[wiki/concepts/rl/RECAP]]：
  - 补充"条件 I 的核心作用"章节：RL 改写为条件监督学习的机制、信号传递路径三方对比（PPO vs AWR vs RECAP）
  - 补充"CFGRL：RECAP 的理论前驱"：$(\star\star)$ 公式来源、贝叶斯翻转路径、$\beta=1$ 退化结论
  - 补充"两种策略改进哲学"：策略优化 vs 条件建模的根本区别
  - 扩展比较表：新增 vs Actor-Critic / PPO / AWR 四维对比
- **更新** wiki/index.md：RL Concepts 4→12 条，AWR 新增行；CFGRL 加入 Knowledge Gaps
- 触发来源：用户关于 AWR / CFGRL / RECAP vs 传统 RL 的三轮 Q&A 对话

## [2026-05-18] update | 补充 BPTT 前置概念到 Flow Q-Learning
- **目标文件**：`wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning.md`
- **新增章节**：## 前置概念：BPTT (Backpropagation Through Time)
  - 算法机制：迭代结构 + 链式法则展开公式
  - 在 Flow / Diffusion 策略中的体现：T 步 ODE 链路穿透
  - 四类瓶颈：显存爆炸 / 梯度病态 / 计算时间 / 数值噪声
  - 三种应对策略对比：DPPO/ReinFlow vs FQL vs 直接 BPTT
- **frontmatter 更新**：tags 增加 `BPTT`；updated 改为 2026-05-18
- **触发来源**：用户在 RECAP 笔记上下文中提问 "BPTT 指的是什么"

## [2026-05-18] query | RL 微调生成模型的技术挑战与解法全景（Analysis 新建）
- **触发来源**：用户请求生成一页 analyses，介绍 RL 微调生成模型的局限性及主流方法的解法
- **新建** [[RL微调生成模型的技术挑战与解决方案]]
  - 系统梳理四大技术障碍：① log-prob 不可解析（完整推导 ODE 积分分解）② BPTT 三重代价（显存/梯度病态/计算时间）③ 探索-利用悖论（Flow ODE 无随机性）④ 离线策略抽取难题
  - 三条主流解法路线：A（Markov 化 PG：DPPO + ReinFlow）/ B（CFM loss ratio：FPO）/ C（离线 Q-learning：FQL + OFQL + FAN）+ 正交优化轴（WarmPrior）
  - 严谨数学推导：log-prob 不可解析 ODE 积分代价 $\mathcal{O}(d^3 T)$；DPPO 两层 MDP 封闭 log-prob；ReinFlow 噪声注入 Markov 化完整步骤；FQL 解耦消除 BPTT 的梯度路径分析
  - 含实用选型矩阵（7 场景）与三条路线根本哲学对比
- **更新** wiki/index.md：Analyses 8→9；新增表格行

## [2026-05-23] ingest+lint+update | 全 vault polish + CLAUDE.md 精简重构
- **MinerU 转换**（6 篇待处理 PDF）
  - `raw/assets/papers/Agent/Yang 等 - 2026 - ARIS ...pdf` → `raw/sources/papers/Agent/Yang 等 - 2026 - ARIS .../`
  - `raw/assets/papers/VLA+RL/Gao - 2026 - FlowRL ...pdf` → `raw/sources/papers/VLA+RL/Gao - 2026 - FlowRL .../`
  - `raw/assets/papers/VLA+RL/Kang - 2026 - WarmPrior ...pdf` → 同上
  - `raw/assets/papers/VLA+RL/McAllister - 2025 - FPO ....pdf` → 同上
  - `raw/assets/papers/VLA+RL/Yi 等 - 2026 - Flow Policy Gradients ...pdf` → 同上（暂无对应 wiki page；列入待补 Source 行列）
  - `raw/assets/papers/VLA+RL/Zhang 等 - 2026 - ReinFlow ...pdf` → 同上
  - 其余 PDF（Lee FAN、Nguyen OFQL、Zhai Critic 等）已先前完成转换，本次 batch `--resume` 跳过。
- **Lint 修复**
  - `wiki/index.md` 统计修正：Entities 13→12、Concepts 12→13（匹配实际文件计数）。
  - `wiki/index.md` 失效链接修复：`[[π₀.₇ 详细解析]]`、`[[π₀.₆ 与 RECAP 原理解析]]`、`[[RL微调生成模型的技术挑战与解决方案]]` → 全部补 `wiki/analyses/` 前缀，并修正末尾的额外 `]`。
  - 公式格式违规修复（伪代码块 → LaTeX 列表）：`wiki/analyses/DDPM & DDIM 完整数学推导.md`（4 处算法块）、`wiki/analyses/DPPO 完整数学推导.md`（1 个算法块）、`wiki/analyses/Flow Matching 完整数学推导.md`（训练伪代码）；将 `ᾱ_t`、`ε_θ`、`θ₀`、`σ_t` 等 Unicode 符号统一改写为 `$\bar{\alpha}_t$`、`$\varepsilon_\theta$`、`$\theta_0$`、`$\sigma_t$`。
  - 代码字串包裹公式修复：`wiki/analyses/UMI ee6d 位姿变换推理.md:237` `` `F_T_NE = I` `` → `${}^{F}T_{NE} = I$` 等 4 处。
- **图片嵌入**（首次启用 §2.3 新规则）
  - `wiki/sources/rl-finetuning/2026-05-17 FlowRL (Gao 2026).md`：嵌入 taxonomy figure。
  - `wiki/sources/rl-finetuning/2026-04-18 ReinFlow.md`：嵌入 Hopper-v2 训练曲线。
  - `wiki/sources/rl-finetuning/2026-05-17 FPO Flow Matching Policy Gradients (McAllister 2025).md`：嵌入 gridworld flow visualization。
  - `wiki/sources/rl-finetuning/2026-05-17 WarmPrior (Kang 2026).md`：嵌入 Figure 1 temporal prior diagram。
  - `wiki/sources/agent-systems/2026-05-16 ARIS.md`：嵌入系统总览 Figure 1，并补充 paper-side raw 链接。
  - 上述 5 个 source 页的 frontmatter `sources` 字段同步改写为 `raw/sources/papers/.../<stem>.md` 路径。
- **CLAUDE.md 重构（v2.0 → v2.1）**
  - 整体压缩：18749 bytes → 7594 bytes（约 59.5% 缩减）；6 节结构保留，§2 / §5 内部合并为表格化速查。
  - 新增 §2.3 图片嵌入规则与 §3.1 步骤 5「选图」。
  - `~/.claude/skills/mineru/skill.md` 同步追加「Image References (Wiki Promotion Rule)」段落。
- **Knowledge gap 标记**
  - Yi 2026《Flow Policy Gradients for Robot Control》：MinerU 已完成转换，wiki Source 页尚未建立；列入下一次扫描时补建。

---

## [2026-05-25] ingest | 4 篇待处理论文：FPO++、LaST-R1、π0-FPO、VLAC

**触发原因**：用户请求更新 wiki；扫描发现 4 篇已完成 MinerU 转换但无 wiki Source 页的论文。

**新建 Source 页（4 个）**：
- [[wiki/sources/rl-finetuning/2026-05-25 FPO++ (Yi 2026)]]（来源 #43）：FPO++ = per-sample CFM ratio clipping + ASPO 非对称 trust region；腿足/仿人/操控机器人控制，sim-to-real 到 Booster T1 + Unitree G1。分类：生成模型+RL（无 LLM 骨干）。
- [[wiki/sources/vla-rl/2026-05-25 LaST-R1 (Chen 2026)]]（来源 #44）：LAPO 联合优化 latent CoT + action；Qwen3-VL-4B + DINOv3 latent targets；自适应 CoT 长度；LIBERO 99.9%、实机 90%。
- [[wiki/sources/vla-rl/2026-05-25 π0-FPO RFT Flow-VLA (Lyu 2025)]]（来源 #45）：π0 VLA 在线 RL 微调；likelihood-free CFM ratio + latent Euler 探索 + Q-ensemble；LIBERO 87.2%，LIBERO-Long 65.3%。**注意**：与 McAllister FPO (#29) 和 Yi FPO++ (#43) 同名但独立开发。
- [[wiki/sources/vla-rl/2026-05-25 VLAC (Zhai 2025)]]（来源 #46）：InternVL actor+critic 统一模型；pairwise progress delta 作 dense reward；graded HIL 三级协议；4 真实任务 30%→90%（200 episodes），HIL 样本效率提升 50%。

**图片嵌入**：
- FPO++ (来源 43)：嵌入 Fig 1 子图（quadruped trotting）
- LaST-R1 (来源 44)：嵌入 Fig 2 LAPO 框架图（确认为清晰架构图）
- π0-FPO (来源 45)：嵌入 ALOHA/LIBERO 任务基准图
- VLAC (来源 46)：嵌入 Fig 1/2 VLAC 架构 + 训练数据图（确认为清晰架构图）

**概念升级提醒**：
- **CFM Loss Ratio 作为 IS Ratio 替代**：来源 #29（McAllister）+ #43（Yi）+ #45（Lyu）→ 已达 ≥3 来源，**待建 Concept 页**（下次 ingest 时执行）

**新增 Knowledge Gaps（9 条）**：
- 来源 43：ASPO、Per-Sample Flow Ratio、Booster T1
- 来源 44：LAPO、Adaptive Latent CoT、DINOv3 CLS Latent Targets
- 来源 45：Multi-step Latent Euler Exploration、Q-Ensemble for Flow-VLA RL
- 来源 46：VLAC 架构、Pairwise Progress Delta Reward、Graded HIL Protocol

**更新 wiki/index.md**：
- Stats：Sources 42→46；VLA+RL 节 7→10 行；生成模型+RL 节 9→10 行
- 新增 CFM Loss Ratio 升级提醒
- 新增 4 个来源的 Knowledge Gaps 块

## [2026-05-25] ingest | SAC Flow (Zhang 2026) + Decoupled Q-Chunking (Li 2025)

**新建 Source 页（2 个）**：
- [[wiki/sources/rl-finetuning/2026-05-25 SAC Flow (Zhang 2026)]]：Flow rollout ≡ Residual RNN → GRU/Transformer 速度重参数化 + noise-augmented rollout 实现端到端 SAC 训练；OGBench offline-to-online +60%、MuJoCo from-scratch +130%
- [[wiki/sources/rl-finetuning/2026-05-25 Decoupled Q-Chunking (Li 2025)]]：首次形式化 action chunking Q-learning 理论（OLC 条件、value bias 上界）；DQC 算法通过 partial critic distillation 解耦 critic/policy 块大小；OGBench 聚合得分 82 vs QC 25

**新建 Concept 页（1 个）**：
- [[wiki/concepts/benchmarks/OGBench]]：长视野目标条件离线 RL 基准（Park et al. 2025a），来源 47+48 同时使用，达 ≥2 阈值

**MinerU 解析**：
- SAC Flow：full.md 成功（zip 解压绕过 Windows MAX_PATH），134 张图片全部提取
- DQC：直接解析成功

**更新 wiki/index.md**：
- Stats：Sources 46→48；Concepts 13→14（新增 OGBench）
- 生成模型+RL 节新增来源 47、48
- Benchmarks 节新增 OGBench
- 新增两篇论文的 Knowledge Gaps（6 条）
- 标记 OGBench 升级完成