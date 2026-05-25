---
type: concept
tags: [Benchmark, OfflineRL, GoalConditioned, LongHorizon, RobotManipulation]
sources: [wiki/sources/rl-finetuning/2026-05-25 Decoupled Q-Chunking (Li 2025), wiki/sources/rl-finetuning/2026-05-25 SAC Flow (Zhang 2026)]
created: 2026-05-25
updated: 2026-05-25
---

# OGBench

**OGBench**（Offline Goal-conditioned RL Benchmark）是 UC Berkeley Seohong Park 等人提出的离线目标条件强化学习基准，专为测试长视野稀疏奖励任务下的 RL 方法而设计。

**参考文献**：Park et al., 2025a（区别于同一作者的 [[wiki/sources/rl-finetuning/2026-04-19 Flow Q-Learning]] 工作）

---

## 设计目标

标准 offline RL 基准（D4RL 等）任务视野较短、奖励较密，难以区分方法在 bootstrapping bias、value backup 设计上的差异。OGBench 专门选取：

- **长视野任务**（large horizon $H$）：使 bootstrapping bias 影响显著
- **稀疏目标奖励**：到达指定目标状态才获得奖励
- **多样化子任务集**：覆盖各种接触精度和操控复杂度的机器人操控场景

---

## 评测场景

OGBench 包含多组机器人操控环境，文献中通常报告"最难 6 个环境"的聚合得分（aggregated score）作为主要指标。

典型任务类别：
- **推/抓/拼接** 等接触密集操控
- **长链式顺序任务**（如多步拼装）
- 不同程度的 goal 多样性（单目标 → 分布式目标）

---

## 已评测方法与结果

（以 [[wiki/sources/rl-finetuning/2026-05-25 Decoupled Q-Chunking (Li 2025)]] 报告为准，6 个最难环境聚合得分）

| 方法 | 得分 |
|------|------|
| **DQC** | **82** |
| NS（n-step return） | 68 |
| SHARSA | 44 |
| QC（Q-chunking，等块） | 25 |
| HIQL | 18 |

[[wiki/sources/rl-finetuning/2026-05-25 SAC Flow (Zhang 2026)]] 在 offline-to-online 设置（在 OGBench 上预训练再在线微调）下，相较 Flow Q-Learning 等基线取得最高 **+60% success rate**。

---

## 在相关工作中的地位

OGBench 是长视野目标条件 offline/offline-to-online RL 方法的事实评测标准，覆盖两类方法：

1. **Value backup 设计**（DQC、n-step return 等）：测试方法是否能有效缓解 bootstrapping bias
2. **表达性策略 + RL**（SAC Flow 等）：测试能否在稀疏奖励 offline-to-online 场景中高效迁移

参见 [[wiki/concepts/benchmarks/LIBERO]] 了解密集奖励、短视野操控基准；两者覆盖互补。
