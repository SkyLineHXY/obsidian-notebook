# Wiki Log

> Compact changelog. Historical detail was compressed on 2026-06-08 to reduce token cost. Use Git history for full pre-compaction text if needed.

## [2026-07-03] ingest | Knowledge Insulation VLA (Driess 2025)

- 摄取 Knowledge Insulation 论文（arXiv 2505.23705，Physical Intelligence，MinerU 转换），新建 [[wiki/sources/vla/2026-07-03 Knowledge Insulation VLA (Driess 2025)]]。
- 核心：诊断"从零初始化 action expert 的梯度污染 VLM backbone → 训练慢 + 语言跟随差 + 冻结不可行"；提出三件套——离散 FAST token 做表示学习信号 + VLM 数据 co-training + attention 内 stop-gradient 切断 action expert→backbone 梯度（含 $P_{bb}/P_{ab}/P_{aa}$ 分块 softmax 与 sg 推导）。结果 LIBERO-90/Spatial SOTA，收敛较 π₀ 快 ~7.5×。
- 更新 index Sources 61→62、vla 节 5→6；Knowledge Insulation 达 ≥2 源（π₀.₅ + 本文），移入 watchlist 升级候选。

---

## [2026-07-02] ingest | π₀.₅ Open-World Generalization + 代码级分析

- 摄取 π₀.₅ 论文（arXiv 2504.16054，MinerU 转换），新建 [[wiki/sources/vla/2026-07-02 π₀.₅ Open-World Generalization (Intelligence 2025)]]，结合用户指定仓库 `cijerezg/lerobot pi05_full` 逐文件分析。
- 重点解析：分层子任务推理（`generate_subtask_tokens` 自回归 + 2s 缓存 + 特殊 token 屏蔽 + BOS 对齐 bug）、离散 FAST + 连续 Flow 混合训练三损失、Knowledge Insulation、自定义注意力掩码、两阶段训练 Pipeline。
- 更新 [[wiki/entities/models/π₀.₅]] 增补 source；index Sources 60→61，vla 节 4→5。

---

## [2026-06-08] maintenance | 压缩 index/log/CLAUDE 以降低上下文成本

- `wiki/index.md` 改为紧凑导航图：主题计数、核心页面入口、近期重点与短 watchlist。
- `wiki/log.md` 由逐项长日志压缩为里程碑时间线；后续每条控制在 1-3 bullets。
- `CLAUDE.md` 简化为硬规则速查，避免与详细 schema 重复。

---

## Compressed History

### 2026-04-18 | init + first ingest
- 建立 `raw/` + `wiki/` + `CLAUDE.md` wiki 架构。
- 摄取 Ubuntu/RT kernel、Diffusion Policy 复现、ReinFlow 等首批材料；建立 Ubuntu、Franka、Diffusion Policy、PREEMPT_RT 等早期实体/概念页。

### 2026-04-19 | RL/VLA/framework expansion
- 摄取 Diffusion Policy、DPPO、Flow Q-Learning、π₀.₆ RECAP、LeRobot、RLinf、StarVLA、Embodied-AI-Guide 等。
- 新建 Flow Matching、DPPO、Offline RL、VLA、RLinf、StarVLA、RoboTwin 等关键页；建立首个 RL 微调对比页。

### 2026-04-21 to 2026-04-25 | math analyses + IL/data collection
- 摄取 DDPM、ACT/ALOHA、DemoSpeedup、VLASH、πRL、HIL-SERL、UMI、FastUMI 等。
- 新建/更新 DDPM & DDIM、Flow Matching、DemoSpeedup、ReinFlow、UMI ee6d 等数学推导与分析页。

### 2026-04-29 to 2026-05-06 | π₀.₇ + UMI refinement
- 摄取并解析 π₀.₇；新建 π₀.₇ entity/source/analysis。
- UMI ee6d 分析重构为纯 EE-at-$t_0$ 方案。

### 2026-05-17 to 2026-05-18 | taxonomy refactor + VLA/RL/lab automation
- 重组 taxonomy：区分 `vla-rl` 与 `rl-finetuning`。
- 摄取 BayesianVLA、5 篇 flow/RL 策略论文、5 篇 VLA-RL 论文、5 篇 Lab Automation 论文。
- 新建/更新 VLA RL 对比、生成模型 RL 挑战分析、π₀.₆/RECAP、AWR 等页面。

### 2026-05-23 | vault polish
- 进行全 vault polish 与 CLAUDE.md 精简重构。
- 强化 frontmatter、公式、图片引用、raw 回链等规范。

### 2026-05-25 to 2026-05-28 | flow/VLA RL late batch
- 摄取 FPO++、LaST-R1、π0-FPO、VLAC、SAC Flow、Decoupled Q-Chunking、Q-Chunking、RL-100、RLT。
- 新建 Action Chunking Q-Learning、Iterative Offline RL、Representation Regularization in Visuomotor RL 等概念页；补充 RLT 架构图与实现细节。

### 2026-06-02 | agentic VLA
- 摄取 Agentic Robot 与 Sci-VLA。
- 新建 Agentic VLA 系统对比页，梳理 SAP 验证-恢复与 Transition Action Insertion 两条长程任务路线。

### 2026-06-07 to 2026-06-08 | SLAM / active perception
- 摄取 X-ICP、GPS-Denied LiDAR SLAM survey、Active SLAM survey、FAST-LIO、FAST-LIO2、张驰洲 2025 主动激光 SLAM 博士论文。
- 新建 SLAM 与 LiDAR-Inertial Odometry 概念页；沉淀 [[wiki/analyses/地下退化环境具身主动感知与自适应导航研究设计]]。
- 研究设计页 v2/v2.1 引入 EARE 稀疏信息图、退化感知节点特征、FAST-LIO2 前端数学。

### 2026-06-08 | GEO spectral-image fusion query
- 应用户请求调研 GEO “谱相融合”特征提取与多模态融合方法。
- 结论：推荐“视觉对象分割 + 对象内部光谱证据提取 + gated/cross-attention fusion + 可解释输出”。

### 2026-07-02 | ingest | RTC (Real-Time Chunking)
- 摄取 Black 2025《Real-Time Execution of Action Chunking Flow Policies》(arXiv 2506.07339, NeurIPS 2025)；MinerU 转换 + 全文精读。
- 新建 [[wiki/sources/vla/2026-07-02 Real-Time Chunking (Black 2025)]]：inpainting 视角的异步动作块执行（冻结前缀 + ΠGDM 软掩码引导）。
- 深入分析 LeRobot `policies/rtc` 源码：`RTCProcessor.denoise_step`（Eq.2/3 符号对应）、`get_prefix_weights`（Eq.5 四种调度）、`ActionQueue`/`LatencyTracker` 异步机制，及 SmolVLA/π₀ 装饰器式集成。index Sources 59→60，vla 3→4。

### 2026-08-08 | ingest | GR00T 系列四代（N1 / N1.5 / N1.6 / N1.7）
- 摄取 GR00T N1 论文（arXiv 2503.14734）+ N1.5/N1.6 官方博客 + N1.7 GitHub README；新建 4 个 source 页、实体页 [[wiki/entities/models/GR00T]]、分析页 [[wiki/analyses/GR00T 系列演进详细解析]]（含四代横向对比总表与 §8 严谨数学推导）。
- 演进主线：VLM 骨干四换（Eagle-2→Eagle 2.5→Cosmos-2B→Cosmos-Reason2-2B）；VLM 冻结策略往复（部分微调→全冻结→顶4层解冻→全冻结）；动作空间 绝对→状态相对→**人机统一相对 EEF**，后者是 20K 小时 EgoScale 人类视频得以直接入训的前提。
- **注意**：N1 论文未经 MinerU 解析（`MINERU_TOKEN` 已过期，API 返回 `user authenticate failed`），改用 PyMuPDF 抽取文本并按图注裁切 3 张图存入 `raw/sources/papers/VLA/<stem>/images/`。Token 续期后建议重新解析。index Sources 62→66，vla 6→10，Entities 12→13，Analyses 10→11。

### 2026-08-08 | ingest | ActiveUMI (Zeng 2025)
- 摄取 arXiv 2510.01607《ActiveUMI: Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations》；新建 [[wiki/sources/data-collection/2026-08-08 ActiveUMI (Zeng 2025)]] 与三方对比页 [[wiki/comparisons/UMI 系列采集接口对比]]（UMI / FastUMI / ActiveUMI）。
- **核心 diff**：VR inside-out 追踪替代 ORB-SLAM3（RPE 10.1→4.0 mm）；把操作员 HMD 的 6-DoF 录进动作空间（20 DoF）使策略可预测视角；为此**被迫放弃 UMI 的相对轨迹表示**改用绝对世界坐标 + 三重标定——头手空间关系无法用相对轨迹表达。吞吐上是净退步（1.49–2.06× 裸手），非帕累托改进。
- **批判点已记入 Caveats**：表中 "UMI" 基线仅是 wrist-only 消融而非真实 UMI 系统；RPE 公式（百分比）与图注单位（mm）矛盾且主语写错；4.2 节 "PourWater" 任务不存在于任何表格；3.1 节整段重复 Related Work；每格仅 10 trials。
- **注意**：`MINERU_TOKEN` 已于 2026-07-20 过期（`user authenticate failed`），继续用 PyMuPDF 抽文本 + 按图区裁切 5 张图。另：raw 解析路径长 266 字符超 Windows MAX_PATH，写入需 `\?\` 前缀。index Sources 66→67，data-collection 2→3，Comparisons 3→4。
