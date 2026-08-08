# Wiki Index

> Compact navigation map. Read this first, then open the relevant page(s). Keep this file short: no full source paths, no long knowledge-gap dumps.

## Snapshot
- **Sources**: 67
- **Entities**: 13
- **Concepts**: 19
- **Comparisons**: 4
- **Analyses**: 11
- **Last compacted**: 2026-06-08

## Source Themes

| Theme                | Count | Focus                                                                                                                              |
| -------------------- | ----: | ---------------------------------------------------------------------------------------------------------------------------------- |
| `agent-systems`      |     3 | ARIS, Agentic Robot, Sci-VLA; agentic long-horizon robotics and research harnesses                                                 |
| `vla-rl`             |    11 | VLA post-training / RL: RECAP, πRL, iRe-VLA, VLA-RL, GR-RL, SA-VLA, VLA-OPD, LaST-R1, π0-FPO, VLAC, RLT                            |
| `rl-finetuning`      |    14 | Diffusion/flow policies + RL: ReinFlow, FQL, DPPO, HIL-SERL, FlowRL, FPO/FPO++, WarmPrior, OFQL, FAN, SAC Flow, Q-Chunking, RL-100 |
| `generative`         |     2 | Flow Matching, DDPM                                                                                                                |
| `imitation-learning` |     3 | Diffusion Policy, ACT/ALOHA, local reproduction notes                                                                              |
| `vla`                |    10 | π₀.₅, VLASH, π₀.₇, BayesianVLA, RTC (Real-Time Chunking), Knowledge Insulation, GR00T N1/N1.5/N1.6/N1.7                            |
| `frameworks`         |     4 | LeRobot, RLinf, StarVLA, RLinf docs                                                                                                |
| `data-collection`    |     3 | UMI, FastUMI, ActiveUMI（VR 头显 + 主动感知）                                                                                             |
| `data-efficiency`    |     1 | DemoSpeedup                                                                                                                        |
| `infrastructure`     |     3 | Ubuntu/RT kernel/controller framework                                                                                              |
| `guides-tools`       |     2 | Embodied-AI-Guide, MinerU Document Explorer                                                                                        |
| `lab-automation`     |     5 | ChemBot, BioMARS, Touch in the Wild, Intelligent Science Laboratory, scaling laws                                                  |
| `slam`               |     6 | GPS-denied LiDAR SLAM, Active SLAM, X-ICP, FAST-LIO/2, 张驰洲 active LiDAR SLAM thesis                                                |

## Core Entities

| Category | Pages |
|---|---|
| Models | [[wiki/entities/models/π₀.₅]], [[wiki/entities/models/π₀.₇]], [[wiki/entities/models/SmolVLA]], [[wiki/entities/models/GR00T]] |
| Frameworks | [[wiki/entities/frameworks/ARIS]], [[wiki/entities/frameworks/RLinf]], [[wiki/entities/frameworks/StarVLA]] |
| Hardware | [[wiki/entities/hardware/Ubuntu 20.04]], [[wiki/entities/hardware/Franka Research 3]], [[wiki/entities/hardware/UMI]] |
| Systems / Tools / People | [[wiki/entities/systems/HIL-SERL]], [[wiki/entities/tools/Claude Code]], [[wiki/entities/people/qq_59001382]] |

## Core Concepts

| Area | Pages |
|---|---|
| RL | [[wiki/concepts/rl/Offline 强化学习]], [[wiki/concepts/rl/RECAP]], [[wiki/concepts/rl/DPPO]], [[wiki/concepts/rl/AWR]], [[wiki/concepts/rl/Action Chunking Q-Learning]], [[wiki/concepts/rl/Iterative Offline RL]], [[wiki/concepts/rl/Representation Regularization in Visuomotor RL]] |
| Imitation Learning | [[wiki/concepts/imitation-learning/ACT]], [[wiki/concepts/imitation-learning/HG-DAgger]] |
| Generative Models | [[wiki/concepts/generative-models/DDPM]], [[wiki/concepts/generative-models/Diffusion Policy]], [[wiki/concepts/generative-models/Flow Matching]] |
| VLA | [[wiki/concepts/vla/Vision-Language-Action 模型]] |
| Benchmarks | [[wiki/concepts/benchmarks/LIBERO]], [[wiki/concepts/benchmarks/OGBench]], [[wiki/concepts/benchmarks/RoboTwin]] |
| Infrastructure | [[wiki/concepts/infrastructure/PREEMPT_RT实时内核]] |
| SLAM | [[wiki/concepts/slam/SLAM (Simultaneous Localization and Mapping)]], [[wiki/concepts/slam/LiDAR-Inertial Odometry]] |

## Comparisons

- [[wiki/comparisons/RL 微调表达性策略方法对比]] — DPPO/ReinFlow/FQL/HIL-SERL/FlowRL/FPO/WarmPrior/OFQL/FAN; log-prob 解法路线与 on/off-policy 对比。
- [[wiki/comparisons/VLA RL 微调方法对比]] — AR-VLA vs Flow-VLA; RECAP/πRL/VLA-RL/GR-RL/SA-VLA/VLA-OPD/iRe-VLA。
- [[wiki/comparisons/Agentic VLA 系统对比]] — Agentic Robot vs Sci-VLA; SAP 验证-恢复 vs 过渡动作插桩。
- [[wiki/comparisons/UMI 系列采集接口对比]] — UMI / FastUMI / ActiveUMI；相对轨迹 vs 绝对坐标、腕视角 vs 主动头、吞吐 vs 精度三大张力。

## Analyses

- [[wiki/analyses/Flow Matching 完整数学推导]]
- [[wiki/analyses/DPPO 完整数学推导]]
- [[wiki/analyses/DDPM & DDIM 完整数学推导]]
- [[wiki/analyses/DemoSpeedup 逐帧熵估计与HDBSCAN聚类数学推导]]
- [[wiki/analyses/ReinFlow π_rl 完整数学推导]]
- [[wiki/analyses/UMI ee6d 位姿变换推理]]
- [[wiki/analyses/π₀.₇ 详细解析]]
- [[wiki/analyses/π₀.₆ 与 RECAP 原理解析]]
- [[wiki/analyses/RL微调生成模型的技术挑战与解决方案]]
- [[wiki/analyses/地下退化环境具身主动感知与自适应导航研究设计]]
- [[wiki/analyses/GR00T 系列演进详细解析]] — N1→N1.5→N1.6→N1.7 四代逐版本 diff、横向对比总表与数学推导。

## Recent Focus

- **SLAM / active perception**: FAST-LIO2 + X-ICP + Active SLAM + 张驰洲 EARE thesis have been synthesized into [[wiki/analyses/地下退化环境具身主动感知与自适应导航研究设计]].
- **Robot policy RL**: flow/diffusion policy RL and VLA RL are split into `rl-finetuning` and `vla-rl`; use the two comparison pages for entry.
- **GEO / spectral-image fusion**: a recent query produced a prose summary in chat, but no durable wiki page currently exists in `wiki/analyses/`.

- **VLA 基础模型两条路线**: NVIDIA GR00T（[[wiki/entities/models/GR00T]]，cross-attention + 数据规模化）与 PI π 系列（[[wiki/entities/models/π₀.₇]]，action expert + RL/prompt conditioning）已可对照阅读。

## Upgrade Watchlist

Keep this short. Add only concepts that are near the ≥2-source threshold or likely to become reusable:

- **Knowledge Insulation (stop-gradient + 离散表示信号 + VLM co-training)**: 现有 π₀.₅ 与 Driess 2025 两源，达阈值，建议升级为 concept 页。
- **CFM Loss Ratio as IS Ratio Substitute**: appears across FPO / FPO++ / π0-FPO; candidate concept page.
- **Inference Scaling / Process Reward for Robotics**: appears in VLA-RL family; watch for second independent source.
- **State Gap / Transition Action Insertion**: Sci-VLA-specific now; candidate if more long-horizon VLA sources arrive.
- **LiDAR Localizability / Partial Localizability**: currently X-ICP-centered; candidate if more SLAM degeneracy sources arrive.

## Maintenance Notes

- To find exact source pages, use directory search under `wiki/sources/<theme>/`.
- Append detailed operation history to [[wiki/log]], but keep new log entries to 1-3 bullets.
- When adding new pages, update only the relevant compact row/section here.
