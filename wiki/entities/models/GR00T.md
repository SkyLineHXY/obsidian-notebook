---
type: entity
tags: [VLA, Foundation Model, Humanoid Robot, NVIDIA, GR00T, Flow Matching, DiT, Cross-Embodiment, Open Source]
sources: [NVIDIA 等 - 2025 - GR00T N1 An Open Foundation Model for Generalist Humanoid Robots, research.nvidia.com/labs/gear/gr00t-n1_5, research.nvidia.com/labs/gear/gr00t-n1_6, github.com/NVIDIA/Isaac-GR00T]
created: 2026-08-08
updated: 2026-08-08
---

# GR00T（NVIDIA Isaac GR00T）

**GR00T** 是 NVIDIA **GEAR Lab** 的开源人形机器人 **VLA（Vision-Language-Action）基础模型系列**，2025 年 3 月发布首版 N1，至 2026 年迭代到 N1.7 并进入 GA（General Availability）。

**项目**：https://developer.nvidia.com/isaac/gr00t ｜ **代码**：https://github.com/NVIDIA/Isaac-GR00T
**许可**：代码 Apache 2.0；权重 NVIDIA Open Model License
**详细解析**：[[wiki/analyses/GR00T 系列演进详细解析]]

---

## 定位

面向**通用人形机器人操作**的跨本体基础模型：接收多模态输入（RGB 图像 + 自然语言指令 + 本体感觉状态），输出连续电机动作；通过 post-training 适配到特定本体、任务与环境。

与 [[wiki/entities/models/π₀.₅]] / [[wiki/entities/models/π₀.₇]]（Physical Intelligence）构成当前开放 VLA 基础模型的两条主要路线。

---

## 不变的架构骨架（四代共享）

| 组件 | 说明 |
|---|---|
| **System 2 — VLM** | 编码图像 + 语言，输出视觉-语言 token $\varphi_t$；低频（N1 约 10 Hz） |
| **System 1 — DiT 动作头** | flow matching 去噪生成 action chunk；高频（N1 约 120 Hz） |
| **耦合方式** | **Cross-attention**（DiT 交叉注意 VLM token）——区别于 π 系列的 action expert / MoE |
| **跨本体机制** | 每本体一套独立 MLP：State Encoder / Action Encoder / Action Decoder；DiT 主干共享。对应代码里的 `--embodiment-tag` |
| **训练目标** | Flow matching，$\mathcal{L}_{\text{fm}} = \mathbb{E}_\tau\big[\|V_\theta(\varphi_t, A_t^\tau, q_t) - (\epsilon - A_t)\|^2\big]$ |
| **数据格式** | LeRobot v2 变体 + GR00T 专有 `meta/modality.json` |

cross-attention 的松耦合设计使 VLM 骨干可整代替换而动作头基本不动——这解释了四代的骨干更换节奏。

---

## 版本速查

| 版本 | 时间 | 参数 | VLM 骨干 | DiT 层 | 动作空间 | 关键改动 |
|---|---|---|---|---:|---|---|
| **N1** | 2025-03 | 2.2B | Eagle-2（SmolLM2+SigLIP-2） | 16 | 绝对 | 双系统架构 + 数据金字塔（潜动作 / 神经轨迹 / DexMimicGen） |
| **N1.5** | 2025-06 | ~3B | Eagle 2.5（grounding 调优） | 16 | 绝对 | **VLM 全冻结** + adapter LayerNorm + **FLARE** 损失 |
| **N1.6** | 2025-12 | ~3B | Cosmos-2B（弹性分辨率） | **32** | **状态相对** | 删 adapter、解冻 VLM 顶 4 层、多机身真机数据 |
| **N1.7** | 2026（GA） | 3B | **Cosmos-Reason2-2B**（Qwen3-VL） | 16 | **统一相对 EEF** | **20K 小时 EgoScale 人类视频** + ONNX/TensorRT + SONIC 全身控制 |

N1.7 其他配置：`select_layer` 12、`tune_top_llm_layers` 0、state/action 维度 132、`action_horizon` 40。

---

## 每代的核心贡献

- **N1**：确立范式。88 小时真机数据 + 827 小时神经轨迹 + 6,500 小时仿真轨迹 + 人类视频（VQ-VAE 潜动作）。真机上用 **10% 数据（42.6%）逼近 Diffusion Policy 全量（46.4%）**。
- **N1.5**：修复语言跟随。真机 GR-1 语言跟随率 **46.6% → 93.3%**，总成功率 43.3% → 83.0%；Unitree G1 后训练 44.0% → 98.8%。
- **N1.6**：多机身与长程。新增 Bimanual YAM / AgiBot Genie-1 / Galaxea R1 Pro / Unitree G1 全身 loco-manipulation 数据；给出六条真机工程经验（相对动作、归一化统计量、DAgger、RTC 等）。
- **N1.7**：数据轴可扩展性。相对 EEF 打通人机本体，EgoScale 证明人类数据存在 **log-linear scaling law**；LIBERO 平均 96.94%，SimplerEnv Fractal 72.5% / Bridge 62.3%。

---

## 支持的本体（N1.7 部分 embodiment tag）

`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`、`LIBERO_PANDA`、`SIMPLER_ENV_WIDOWX`、`SIMPLER_ENV_GOOGLE`、`ROBOCASA_GR1_TABLETOP`、`UNITREE_G1` / `UNITREE_G1_SONIC`、`NEW_EMBODIMENT`（自定义机器人）。

---

## 已知短板

1. **长程组合任务**：需"先开容器再放入"的两阶段任务成功率极低（SimplerEnv `put_eggplant_in_sink` 2.0%、`place_in_closed_drawer` 7.0%）。
2. **多任务语言跟随与 OOD 泛化**：N1.6 官方自陈仍未达鲁棒泛化。
3. **相对动作在小数据下误差累积**：N1.6 明确提示，尚无系统解法。
4. **无 RL 后训练环节**：四代均停留在模仿学习 + 数据规模化，对比 [[wiki/concepts/rl/RECAP]] 路线。
5. **跨代量化对照缺失**：N1.6 与 N1.7 均未公开与前代的逐 benchmark 对照数据。

---

## 关联

- 详细解析：[[wiki/analyses/GR00T 系列演进详细解析]]
- 源页面：[[wiki/sources/vla/2026-08-08 GR00T N1 (NVIDIA 2025)]] | [[wiki/sources/vla/2026-08-08 GR00T N1.5 (NVIDIA 2025)]] | [[wiki/sources/vla/2026-08-08 GR00T N1.6 (NVIDIA 2025)]] | [[wiki/sources/vla/2026-08-08 GR00T N1.7 (NVIDIA 2026)]]
- 对照模型：[[wiki/entities/models/π₀.₅]] | [[wiki/entities/models/π₀.₇]] | [[wiki/entities/models/SmolVLA]]
- 概念：[[wiki/concepts/vla/Vision-Language-Action 模型]] | [[wiki/concepts/generative-models/Flow Matching]] | [[wiki/concepts/benchmarks/LIBERO]]
- 框架：[[wiki/sources/frameworks/2026-04-19 LeRobot]]
