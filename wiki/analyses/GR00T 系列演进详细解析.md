---
type: analysis
tags: [VLA, Foundation Model, Humanoid Robot, NVIDIA, GR00T, Flow Matching, DiT, Cross-Embodiment, Relative Action, Human Video]
sources: [NVIDIA 等 - 2025 - GR00T N1 An Open Foundation Model for Generalist Humanoid Robots, research.nvidia.com/labs/gear/gr00t-n1_5, research.nvidia.com/labs/gear/gr00t-n1_6, github.com/NVIDIA/Isaac-GR00T, arXiv 2602.16710 EgoScale]
created: 2026-08-08
updated: 2026-08-08
---

# GR00T 系列演进详细解析（N1 → N1.5 → N1.6 → N1.7）

> **问题**：NVIDIA GR00T 系列到底在做什么？每一代版本相对上一代改了哪些东西，为什么这样改？

**关联页面**：[[wiki/entities/models/GR00T]] | [[wiki/sources/vla/2026-08-08 GR00T N1 (NVIDIA 2025)]] | [[wiki/sources/vla/2026-08-08 GR00T N1.5 (NVIDIA 2025)]] | [[wiki/sources/vla/2026-08-08 GR00T N1.6 (NVIDIA 2025)]] | [[wiki/sources/vla/2026-08-08 GR00T N1.7 (NVIDIA 2026)]] | [[wiki/concepts/generative-models/Flow Matching]] | [[wiki/concepts/vla/Vision-Language-Action 模型]]

---

## 0. 一句话总览

GR00T 是 NVIDIA GEAR Lab 的**开源人形机器人 VLA 基础模型系列**，四代共享同一套骨架——**System 2（VLM 编码视觉+语言）+ System 1（DiT flow-matching 动作头）+ 机器人独立的 state/action 投影层**——而每一代的迭代几乎都发生在四条正交轴上：**① 换 VLM 骨干、② 改 VLM 冻结策略、③ 换动作空间表征、④ 换数据来源**。

| 版本       | 时间       | 一句话定位                                                              |
| -------- | -------- | ------------------------------------------------------------------ |
| **N1**   | 2025-03  | 立骨架：双系统 VLA + 数据金字塔（人类视频/仿真/神经轨迹/真机）                               |
| **N1.5** | 2025-06  | 修语言跟随：冻结 VLM + 换更强 grounding 骨干 + FLARE 世界建模目标                     |
| **N1.6** | 2025-12  | 上规模与多机身：Cosmos VLM + 2× DiT + **相对动作** + 多本体真机数据                   |
| **N1.7** | 2026（GA） | 押注人类视频：Cosmos-Reason2 + **统一相对 EEF 空间** + 20K 小时 EgoScale 人类第一视角数据 |

**演进主线的一句话概括**：从「靠合成数据放大真机数据」（N1 的 DreamGen / DexMimicGen）逐步转向「靠人类第一视角视频作为可预测的监督源」（N1.7 的 EgoScale），中间通过**统一的相对末端执行器动作表征**打通了人与机器人两种本体。

---

## 1. 四代不变的骨架

在读迭代内容之前，先明确**什么没变**——这是理解 GR00T 系列的锚点。

### 1.1 双系统架构（Dual-System）

![[raw/sources/papers/VLA/NVIDIA 等 - 2025 - GR00T N1 An Open Foundation Model for Generalist Humanoid Robots/images/fig2_model_overview.png]]
*Figure: GR00T N1 模型总览。图像观测与语言指令 token 化后送入 VLM 骨干（System 2）；VLM 输出连同机器人状态、动作编码送入 Diffusion Transformer（System 1）生成电机动作。*

- **System 2 = VLM**：编码图像 + 语言指令，输出视觉-语言 token $\varphi_t$。低频（N1 中约 10 Hz）。
- **System 1 = DiT 动作头**：以 flow matching 去噪的方式，从噪声生成 action chunk。高频（N1 中约 120 Hz）。

这个命名借用 Kahneman 的快/慢思考，但工程上的实质是：**语义理解算一次，动作采样算多次**。

### 1.2 Flow Matching 动作生成

四代都用 flow matching（而非 DDPM 式扩散）训练动作头，损失形式在 N1 论文中给定（见 §8 数学推导），后续版本未见改动公告。

### 1.3 跨本体（Cross-Embodiment）机制

不同机器人的状态/动作维度不同，GR00T 的解法自始至终是：**每个本体一套独立的 MLP 投影层**（State Encoder / Action Encoder / Action Decoder），中间的 DiT 主干共享。

这就是代码里 `--embodiment-tag` 的由来：tag 决定用哪套 modality config（state/action key 映射 + 归一化统计量）。

### 1.4 与 π 系列的架构分野

| | GR00T | π 系列（[[wiki/entities/models/π₀.₅]] / [[wiki/entities/models/π₀.₇]]） |
|---|---|---|
| VLM↔动作头耦合方式 | **Cross-attention**（DiT 交叉注意 VLM token） | **Mixture-of-Experts / Action Expert**（共享注意力，分离权重） |
| 论文自述理由 | cross-attention 让 VLM 与动作头架构可各自替换，灵活性更高 | action expert 与 VLM 共享 KV，信息带宽更大 |

这个差异解释了为什么 GR00T 能在四代里**反复整体更换 VLM 骨干**（Eagle-2 → Eagle 2.5 → Cosmos-2B → Cosmos-Reason2-2B）而动作头基本不动——耦合面窄，替换成本低。

---

## 2. GR00T N1（2025-03，arXiv 2503.14734）

### 2.1 架构参数

![[raw/sources/papers/VLA/NVIDIA 等 - 2025 - GR00T N1 An Open Foundation Model for Generalist Humanoid Robots/images/fig3_architecture.png]]
*Figure: GR00T N1 架构。Eagle-2 VLM（冻结）输出的视觉-语言 token 通过 cross-attention 注入 DiT；State/Action Encoder 与 Action Decoder 为具身相关模块（绿色）；DiT 内部为 cross-attention 与 self-attention 交替堆叠，推理时迭代 K 次。*

| 组件           | 细节                                                                              |
| ------------ | ------------------------------------------------------------------------------- |
| 总参数          | **2.2B**（公开 checkpoint 名 `GR00T-N1-2B`）                                         |
| VLM 骨干       | **Eagle-2**，1.34B；由 **SmolLM2**（LLM）+ **SigLIP-2**（视觉编码器）微调而来                   |
| 图像处理         | 224×224 → pixel shuffle → **每帧 64 个 image token**                               |
| 取哪一层特征       | LLM **第 12 层**（中间层），论文实测中间层比最后一层**推理更快且成功率更高**                                  |
| 动作头          | DiT 变体，**cross-attention 与 self-attention 交替**（类 Flamingo / VIMA），adaLN 注入去噪步条件 |
| Action chunk | $H = 16$                                                                        |
| 去噪步数         | $K = 4$（forward Euler）                                                          |
| 推理耗时         | L40 GPU、bf16，采样 16 步动作 **63.9 ms**                                              |
| 训练算力         | 最多 1024×H100；预训练约 **50,000 H100 GPU·hours**                                     |

**注意力结构细节**：self-attention 作用于「噪声动作 token $A_t^\tau$ + 状态嵌入 $q_t$」，cross-attention 负责对 VLM token $\varphi_t$ 的条件化。最后 $H$ 个 token 过具身相关的 Action Decoder MLP 输出动作。

### 2.2 数据金字塔（本代最大贡献）

![[raw/sources/papers/VLA/NVIDIA 等 - 2025 - GR00T N1 An Open Foundation Model for Generalist Humanoid Robots/images/fig1_data_pyramid.png]]
*Figure: 机器人基础模型训练的数据金字塔。底层为网络数据与人类第一视角视频，中层为仿真轨迹与神经轨迹（合成数据），顶层为真机遥操作数据。*

| 层级           | 内容                                                                                    | 规模                                        |
| ------------ | ------------------------------------------------------------------------------------- | ----------------------------------------- |
| **顶（真机）**    | 内部 Fourier GR-1 遥操作（VIVE Ultimate Tracker 腕部 + Xsens Metagloves 手指，IK 重定向，20 Hz）      | **88 小时**                                 |
|              | Open X-Embodiment（RT-1 / Bridge-v2 / Language Table / DROID / MUTEX / RoboSet / Plex） | —                                         |
|              | AgiBot-Alpha                                                                          | **140,000 条轨迹**                           |
| **中（合成）**    | **仿真轨迹**：DexMimicGen 自动扩增                                                             | **780,000 条 ≈ 6,500 小时，仅耗时 11 小时生成**      |
|              | **神经轨迹**：微调 image-to-video 模型生成反事实轨迹                                                  | **~300k 视频 ≈ 827 小时**（把 88 小时真机数据放大 ~10×） |
| **底（人类/网络）** | Ego4D、Ego-Exo4D、Assembly-101、EPIC-KITCHENS、HOI4D、HoloAssist、RH20T-Human               | —                                         |

### 2.3 两个关键的「无动作标签」解法

底层与中层数据都缺真值动作，N1 给了两条路：

**(a) Latent Actions（潜动作，源自 LAPA）**
训练一个 VQ-VAE：编码器吃 $(x_t,\ x_{t+H})$ 帧对输出潜动作 $z_t$，解码器由 $(z_t, x_t)$ 重建 $x_{t+H}$。训练完只取编码器当作**逆动力学模型**，用**量化前的连续嵌入**作为动作标签，并把它当成一个独立的 "LAPA" 本体，用同一套 flow matching 损失训练。

> 关键效果：所有异构数据共享同一个潜动作空间。论文 Fig.4 展示了「右臂左移」这一潜动作在 8 种本体（含人类）上检索到语义一致的画面。

**(b) IDM 伪标注**
在真机数据上训一个逆动力学模型（IDM），给神经轨迹打伪动作标签。

**实验结论（RoboCasa）**：低数据（30 demo）时 LAPA 略优于 IDM；数据增多（100/300）后 IDM 反超且差距拉大——因为 IDM 训练数据变多后伪标签越来越贴近真实动作。

### 2.4 实验结果

**仿真（每任务 100 demo，成功率）**

| 方法 | RoboCasa | DexMG | GR-1 Tabletop | 平均 |
|---|---:|---:|---:|---:|
| BC-Transformer | 26.3% | 53.9% | 16.1% | 26.4% |
| Diffusion Policy | 25.6% | 56.1% | 32.7% | 33.4% |
| **GR00T-N1-2B** | **32.1%** | **66.5%** | **50.0%** | **45.0%** |

**真机 GR-1（成功率）**

| 方法 | Pick-and-Place | Articulated | Industrial | Coordination | 平均 |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy (10% 数据) | 3.0% | 14.3% | 6.7% | 27.5% | 10.2% |
| Diffusion Policy (全量) | 36.0% | 38.6% | 61.0% | 62.5% | 46.4% |
| GR00T-N1-2B (10% 数据) | 35.0% | 62.0% | 31.0% | 50.0% | **42.6%** |
| **GR00T-N1-2B (全量)** | **82.0%** | **70.9%** | **70.0%** | **82.5%** | **76.8%** |

> **最有说服力的一行**：GR00T-N1 用 **10% 数据**（42.6%）只比 Diffusion Policy 用**全量数据**（46.4%）低 3.8 个点。这是「预训练买数据效率」的直接证据。

**神经轨迹协同训练增益**：RoboCasa 在 30/100/300 demo 下分别 **+4.2% / +8.8% / +6.8%**；真机 GR-1 八任务平均 **+5.8%**。协同训练采样比 1:1。

### 2.5 N1 自陈的局限（这正是后三代的路线图）

论文 §4.6 明说：目前只覆盖**短程桌面操作**；未来需要 **long-horizon loco-manipulation**、**更强的 VL 骨干**（提升空间推理与语言理解）、以及**更好的合成数据物理一致性**。

对照后续版本：N1.5 补语言理解，N1.6 补 loco-manipulation 与多机身，N1.7 补数据源。**路线图执行得相当忠实。**

---

## 3. GR00T N1.5（2025-06）

N1.5 的主题是 **语言跟随（language following）与泛化**，三处改动：

### 3.1 架构：把 VLM 彻底冻住

| 改动 | N1 | N1.5 |
|---|---|---|
| VLM 训练状态 | 预训练/微调阶段部分参与训练（语言部分冻结） | **预训练与微调阶段全程冻结** |
| 视觉→LLM 的 adapter | 原 MLP | **简化的 adapter MLP**，并对送入 LLM 的**视觉与文本 token 各加一层 LayerNorm** |

官方说法：这两处改动「极大改善了语言跟随与泛化」。

> **解读**：这是典型的 **knowledge insulation**（知识隔离）思路——动作头的梯度会污染 VLM 的语义表征，导致模型「只会做动作，不看指令」。冻结 + LayerNorm 稳定输入分布，等于把 VLM 的语义能力保护起来。同期 π 系列在 [[wiki/sources/vla/2026-07-03 Knowledge Insulation VLA (Driess 2025)]] 中给出了同方向但不同手段（stop-gradient + 离散动作 token 辅助信号）的解法。

### 3.2 VLM 骨干：Eagle 2.5 + grounding 专项调优

从 Eagle 2.5 出发，针对 **grounding 与物理理解**做了微调：

| 模型 | 参数 | GR-1 grounding IoU ↑ | RefCOCOg-val IoU ↑ |
|---|---|---:|---:|
| Qwen2.5-VL | 3B | 35.5 | 85.2 |
| **GR00T N1.5 VLM** | 2.1B | **40.4** | **89.6** |

### 3.3 训练目标：加入 FLARE

在 flow matching 损失之外，增加 **FLARE（Future LAtent Representation Alignment）**：不去生成式地建模未来帧，而是**把模型表征对齐到未来的目标嵌入**。

- 效果一：策略性能提升；
- 效果二（更重要）：**解锁了从人类第一视角视频直接学习的能力**——因为对齐目标是"未来的表征"，不需要动作标签。
- 损失系数：预训练与后训练均为 **0.2**。

> 这一步在系列演进中意义重大：N1 时代人类视频要靠 VQ-VAE 潜动作绕道进入训练，N1.5 之后有了更直接的通道；到 N1.7 索性把人类视频推到 20K 小时的量级。

### 3.4 训练配置与数据

- **250K 步，1K 张 H100，global batch size 16384**，AdamW + cosine schedule，warmup ratio 0.05。
- 预训练混合：内部 GR-1 数据、OpenXE、仿真 GR-1（DexMG）、**DreamGen 神经轨迹**、AgiBot-Beta。

### 3.5 实验结果（全面碾压 N1）

**架构验证（from scratch，隔离架构改动的贡献）**

| Benchmark         | N1 (scratch) | N1.5 (scratch) |
| ----------------- | -----------: | -------------: |
| Language Table    |        52.8% |      **93.2%** |
| Sim GR-1 Language |        36.4% |      **54.4%** |

**低数据后训练**

| Benchmark | N1 | N1.5 |
|---|---:|---:|
| RoboCasa, 30 demo/任务 | 17.4 | **47.5** |
| Sim GR-1, 0-shot | 39.6 | **43.9** |
| Sim GR-1, 30 demo/任务 | 43.2 | **47.4** |

**真机 GR-1 语言跟随**（桌上两种水果，指定其一放盘子里；目标水果左右手各 50% 概率）

| 指标 | N1 | N1.5 |
|---|---:|---:|
| 语言跟随率 | 46.6% | **93.3%** |
| 总成功率 | 43.3% | **83.0%** |

> 这张表是理解 N1 缺陷的钥匙：N1 **几乎总能把某个水果放进盘子**，但**不听指令挑哪一个**——即"会动作、不看语言"。46.6% 基本等于随机二选一。

**新物体泛化（10 个预训练未见物体）**

| 设置 | N1 | N1.5 |
|---|---:|---:|
| 0-shot | 0% | **15.0%** |
| FLARE 在含新物体的人类视频上后训练 | — | **55.0%** |

**新动词泛化（DreamGen 12 个新动词）**：N1 **13.1%** → N1.5 **38.3%**。

**Unitree G1 后训练（1K 遥操作 episode）**

| 设置 | 成功率 |
|---|---:|
| N1，2选1（4 种已见水果） | 44.0% |
| **N1.5，2选1（4 种已见水果）** | **98.8%** |
| N1.5，2选1（5 种**未见**物体） | 84.2% |

---

## 4. GR00T N1.6（2025-12）

N1.6 的主题是 **规模化 + 多机身 + 复杂长程任务**。

### 4.1 架构改动（四处）

| # | 改动 | 从 | 到 |
|---|---|---|---|
| 1 | Base VLM | Eagle 2.5 系 | **NVIDIA Cosmos-2B VLM 内部变体**——支持**弹性分辨率**，可按**原生宽高比编码图像、无需 padding**；在通用视觉-语言任务 **和** 具身推理任务（如 next action prediction）上训练 |
| 2 | DiT 规模 | 16 层 | **32 层（2×）** |
| 3 | VLM 后接结构 | N1.5 的 4 层 transformer adapter | **删除 adapter**，改为**预训练阶段解冻 VLM 顶部 4 层** |
| 4 | 动作空间 | 绝对关节角 / 绝对 EEF 位姿 | **多数本体改为「状态相对」的 action chunk** |

> **改动 3 值得单独看**：N1.5 用「冻结 VLM + 外挂 adapter」适配，N1.6 认为不如「去掉 adapter + 让 VLM 顶部几层自己适应」。这是**冻结策略的第一次反转**（N1.7 又会反转回去，见 §5.2）。

### 4.2 数据：从「单一 GR-1」走向「多机身」

在 N1.5 混合数据之上，新增数千小时遥操作数据：

- **Bimanual YAM arms**（双臂）
- **AGIBot Genie-1**
- **Galaxea R1 Pro 仿真**（BEHAVIOR 任务套件）
- **Unitree G1 全身 loco-manipulation**

这正面回应了 N1 论文提出的 "long-horizon loco-manipulation" 缺口。

### 4.3 训练配置

- 预训练 **300K 步，global batch size 16384**。
- 后训练：小规模任务数据集，典型 **10K–30K 步，global batch ≤ 1K**。

### 4.4 官方总结的六条工程经验（这部分实操价值最高）

1. **相对动作是多数本体的默认动作空间**——运动更平滑更准；但**小数据集下相对动作容易误差累积**，损害纠错能力。
2. **归一化统计量的选择**：任务分布与预训练相近时，用**预训练统计量**更好；分布差异大时，用预训练统计量会**欠拟合**，应改用后训练统计量。
3. **N1.6 比 N1.5 收敛更快、动作更平滑，但更容易过拟合**。对策：更强的状态正则、额外数据增强、与预训练数据协同训练。
4. **DAgger 有效**——真机表现不佳时推荐使用。参见 [[wiki/concepts/imitation-learning/HG-DAgger]]。
5. **RTC（Real-Time Chunking）在训练期和测试期都能提升**异步推理下的动作平滑度与鲁棒性，已用于 Unitree G1 与 Bimanual YAM 实验。参见 [[wiki/sources/vla/2026-07-02 Real-Time Chunking (Black 2025)]]。
6. **多任务语言跟随与 OOD 泛化仍然是难题**——更细粒度的子任务标注能改善语言跟随，但**距离鲁棒泛化仍有距离**。

> 第 6 条是罕见的诚实自述：N1.5 在「两个水果二选一」上刷到 93.3%，但那是**受控的单步语言消歧**；一旦上到多任务长程场景，语言跟随问题重新出现。

**N1.6 未公开与 N1.5 的逐 benchmark 对照表**——官方博客只给了定性结论「在仿真基准与真机 YAM / AgiBot Genie-1 / Unitree G1 上均优于 N1.5」。

---

## 5. GR00T N1.7（2026，General Availability）

N1.7 是系列首个 **GA（正式可用）** 版本：提供商业支持与稳定性保证，代码 Apache 2.0，权重走 NVIDIA Open Model License。

### 5.1 配置层面的精确 diff（来自仓库 config 默认值）

| 配置项 | N1.6 | N1.7 |
|---|---|---|
| 模型包命名空间 | `gr00t_n1d6` | `gr00t_n1d7` |
| VLM 骨干 | `nvidia/Eagle-Block2A-2B-v2`（vendored Eagle） | **`nvidia/Cosmos-Reason2-2B`**（Qwen3-VL 架构，**gated 仓库**） |
| transformers | 4.51.3 | 4.57.3 |
| `select_layer` | 16 | **12** |
| `tune_top_llm_layers` | 4 | **0** |
| `load_bf16` | true | **false** |
| state / action 维度 | 29 | **132** |
| `action_horizon` | 16 | **40** |
| DiT 扩散层数 | 32 | **16** |
| 数据集输入 | 单路径 | **多路径 + `ds_weights_alpha` 混合权重** |
| rollout CLI | `--action-horizon` | `--execution-horizon`（澄清"每次策略调用实际执行几步"） |

> ⚠️ **一处需要注意的资料冲突**：HuggingFace 上的 N1.7 介绍博客称动作头为「32 层 DiT」，且 model card 里还写着「SigLip2 视觉编码器 + T5 语言编码器」。前者与仓库 config（16 层）矛盾，后者明显是沿用旧模板的陈述（N1.7 骨干是 Cosmos-Reason2-2B）。**以仓库 config 与 README 的 diff 为准。**

### 5.2 两个核心叙事

**(1) 统一的相对末端执行器动作空间（Relative EEF）**

N1.7 采用**在机器人本体与人类本体之间共享**的相对 EEF 动作空间：动作表示为**相对当前位姿的增量**而非绝对目标。官方明确称这是跨本体性能的**关键因素**。

> **为什么这一步是打通人类视频的前提**：人手和机械臂的绝对关节空间完全无法对齐，但「相对当前末端位姿移动多少」这个量在两种本体上是**同构**的。N1.6 引入相对动作时还只是为了动作平滑，到 N1.7 它变成了**跨本体迁移的载体**。

**(2) 20K 小时 EgoScale 人类视频预训练**

底层研究是 **EgoScale**（arXiv 2602.16710，NVIDIA + UC Berkeley + UMD）：

- 在 **20,854 小时**有动作标注的第一视角人类视频上训练 VLA，规模是既往工作的 **20 倍以上**；
- 发现**人类数据规模与验证损失之间的 log-linear scaling law**，且验证损失**强相关于下游真机性能**——这把大规模人类数据确立为**可预测的监督源**；
- 两阶段迁移配方：**大规模人类预训练 → 轻量的人-机对齐 mid-training**；
- 在 **22-DoF 灵巧手**上相对无预训练基线**平均成功率 +54%**，并能迁移到更低 DoF 的手。

> 官方口径：从 1K 小时到 20K 小时人类数据，**平均任务完成度翻倍以上**。这是机器人灵巧性方向首个被明确宣称的 scaling law。

N1.7 的定位因此是：**性能与 N1.6 相当，但泛化与语言跟随更好**——即这一代买的不是 benchmark 分数，而是**数据轴上的可扩展性**。

### 5.3 其他工程增量

- **部署**：全流程 **ONNX + TensorRT 导出**；覆盖 dGPU / Jetson AGX Thor / Orin / DGX Spark。H100 上 PyTorch eager 85.8 ms（11.7 Hz）→ TensorRT 27.9 ms（35.9 Hz），**3.08× 加速**。
- **全身控制（SONIC）**：通过 `UNITREE_G1_SONIC` tag 接 GEAR-SONIC 控制器；VLA 预测**紧凑的隐动作 token**，由学习到的全身控制器解码为含腿、臂、手的全身关节指令，单策略端到端产生语言条件下的协同操作 + 移动。
- **生态**：已并入 HuggingFace LeRobot（`groot` policy type）。参见 [[wiki/sources/frameworks/2026-04-19 LeRobot]]。
- **训练技巧**：`--state_dropout_prob`（model config 默认 0.8，finetune CLI 默认 0.2）随机丢弃状态输入以降低对本体感觉的依赖；各 benchmark 覆写不同（LIBERO-Long 0.2 / SimplerEnv Bridge 0.8 / Fractal 0.5）。强依赖本体感觉的任务应调低。
- 官方提示：**图像增强非确定性会带来 5–6% 的跑间方差**，对标 benchmark 时需注意。

### 5.4 N1.7 公开 benchmark 成绩

**LIBERO**（微调 checkpoint，四套件同超参，`--state-dropout-prob 0.2`，20K 步，batch 640）

| 套件 | 成功率 |
|---|---:|
| Spatial | 195/200 (97.65%) |
| Goal | 195/200 (97.5%) |
| Object | 197/200 (98.45%) |
| 10 (Long) | 189/200 (94.35%) |
| **平均** | **96.94%** |

**SimplerEnv — Bridge (WidowX)**：平均 **62.3%**

| 任务 | 成功率 |
|---|---:|
| open_drawer | 100.0% |
| close_drawer | 97.0% |
| spoon_on_towel | 78.0% |
| carrot_on_plate | 58.0% |
| put_eggplant_in_basket | 53.0% |
| stack_cube | 48.0% |
| put_eggplant_in_sink | 2.0% |

**SimplerEnv — Fractal (Google Robot)**：平均 **72.5%**

| 任务 | 成功率 |
|---|---:|
| pick_coke_can | 100.0% |
| move_near | 100.0% |
| pick_object | 94.0% |
| close_drawer | 69.0% |
| open_drawer | 65.0% |
| place_in_closed_drawer | 7.0% |

**RoboCasa GR1 Tabletop**（24 任务，每任务 20 试）：平均 **44.5%**；最高 PosttrainPnPNovelFromPlateToPlateSplitA 75.0%，最低 PosttrainPnPNovelFromCuttingboardToBasket 10.0%。该 tag 用 **8 步 action target** 以匹配闭环评测设置。

> **两个尾部任务值得注意**：`put_eggplant_in_sink` 2.0% 与 `place_in_closed_drawer` 7.0%。它们都需要**先打开容器再放入**的两阶段序列——与 N1.6 自述的"多任务长程语言跟随仍是难题"完全吻合。**平均分掩盖了长程组合任务上的系统性短板。**

---

## 6. 四代横向对比总表

| 维度                  | N1 (2025-03)                                           | N1.5 (2025-06)                             | N1.6 (2025-12)                                          | N1.7 (2026 GA)                      |
| ------------------- | ------------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------- | ----------------------------------- |
| **总参数**             | 2.2B                                                   | ~3B                                        | ~3B                                                     | 3B                                  |
| **VLM 骨干**          | Eagle-2 (1.34B)<br/>= SmolLM2 + SigLIP-2               | Eagle 2.5 系（2.1B）<br/>grounding 专项调优       | Cosmos-2B 内部变体                                          | Cosmos-Reason2-2B<br/>(Qwen3-VL)    |
| **图像处理**            | 224×224 固定，pixel shuffle→64 token/帧                    | 同 N1                                       | **弹性分辨率，原生宽高比，无 padding**                               | 弹性分辨率                               |
| **取特征层**            | 第 12 层                                                 | —                                          | 第 16 层                                                  | **第 12 层**                          |
| **VLM 冻结策略**        | 语言部分冻结，其余微调                                            | **全程冻结**                                   | **顶部 4 层解冻**                                            | **全程冻结**（`tune_top_llm_layers=0`）   |
| **VLM→DiT 适配**      | cross-attention                                        | 简化 adapter MLP<br/>+ 视觉/文本 token LayerNorm | **删除 4 层 adapter**                                      | —                                   |
| **DiT 层数**          | 16                                                     | 16                                         | **32**                                                  | **16**                              |
| **Action horizon**  | 16                                                     | 16                                         | 16                                                      | **40**                              |
| **state/action 维度** | —                                                      | —                                          | 29                                                      | **132**                             |
| **动作空间**            | 绝对关节角 / 绝对 EEF                                         | 绝对                                         | **状态相对 action chunk**                                   | **统一相对 EEF（人+机共享）**                 |
| **训练目标**            | Flow matching                                          | Flow matching **+ FLARE**（系数 0.2）          | flow matching (+FLARE)                                  | flow matching                       |
| **预训练规模**           | ~50K H100·h，1024 GPU                                   | **250K 步，1K×H100，bs 16384**                | **300K 步，bs 16384**                                     | —                                   |
| **核心数据增量**          | 数据金字塔：88h 真机 + 827h 神经轨迹 + 6500h 仿真 + 人类视频（VQ-VAE 潜动作） | DreamGen 神经轨迹 + AgiBot-Beta                | **多机身真机**：YAM / AgiBot Genie-1 / Galaxea R1 Pro / G1 全身 | **20K 小时 EgoScale 人类第一视角视频**        |
| **无标签数据解法**         | VQ-VAE 潜动作（LAPA）+ IDM 伪标签                              | **FLARE 未来表征对齐**                           | —                                                       | **统一相对 EEF 直接对齐**                   |
| **主打能力**            | 数据效率（10% 数据≈基线全量）                                      | **语言跟随**（46.6%→93.3%）+ 新物体/新动词泛化           | 多机身、双臂、loco-manipulation、长程                             | 泛化 + 语言跟随 + **可扩展的人类数据轴**           |
| **部署**              | —                                                      | —                                          | RTC 异步推理                                                | **ONNX + TensorRT 全流程**（H100 3.08×） |
| **许可 / 状态**         | 研究预览                                                   | 研究预览                                       | 研究预览                                                    | **GA**，代码 Apache 2.0                |

---

## 7. 三条演进主线的解读

### 7.1 主线一：VLM 骨干的四次更换，方向是「具身推理」

```
Eagle-2 (SmolLM2 + SigLIP-2)     通用 VLM，纯视觉-语言预训练
      ↓  N1.5：grounding 才是瓶颈
Eagle 2.5 + grounding 专项微调    RefCOCOg 89.6 / GR-1 grounding 40.4
      ↓  N1.6：分辨率与具身推理
Cosmos-2B（弹性分辨率）           训练目标含 next action prediction
      ↓  N1.7：推理能力
Cosmos-Reason2-2B (Qwen3-VL)      "Reason" 命名即目标
```

**可提炼的判断**：GR00T 的每一次骨干更换都不是「换个更大的通用 VLM」，而是**换一个在具身相关维度（grounding → 原生分辨率 → 具身推理 → 显式推理）上更强的 VLM**。这与 cross-attention 的松耦合设计互为因果——正因为替换便宜，才敢每代都换。

### 7.2 主线二：VLM 冻结策略的「反复横跳」，其实是在找平衡点

| 版本 | 策略 | 动机 |
|---|---|---|
| N1 | 语言部分冻结，视觉与其余微调 | 默认做法 |
| N1.5 | **全冻结** + adapter + LayerNorm | 动作梯度污染语义表征 → 语言跟随崩坏 |
| N1.6 | 删 adapter，**解冻顶部 4 层** | 全冻结导致 VLM 无法适应具身分布 |
| N1.7 | **回到全冻结**（`tune_top_llm_layers=0`） | 骨干本身已在具身推理上预训练过，无需再适应 |

**这不是摇摆，而是骨干能力提升后最优点的迁移**：当 VLM 本身就在具身数据上训练过（N1.6/N1.7 的 Cosmos 系列含 next-action-prediction 目标），下游就不再需要解冻——把适应工作**前移到了 VLM 预训练阶段**。这与 [[wiki/sources/vla/2026-07-03 Knowledge Insulation VLA (Driess 2025)]] 的核心论点一致：VLA 训练中应当保护 VLM 的语义能力，代价由别处支付。

### 7.3 主线三：动作表征演化是为数据来源服务的

```
N1/N1.5  绝对关节角 / 绝对 EEF
          → 每个本体一套独立空间，人类视频只能靠 VQ-VAE 潜动作"绕道"
N1.6     状态相对 action chunk
          → 动机是动作平滑与精度；副作用：小数据下误差累积
N1.7     统一相对 EEF（人 + 机共享）
          → 动机变成"让人类视频与机器人数据落在同一个动作空间"
```

**最关键的洞察**：N1.7 的相对 EEF 不是一个孤立的工程优化，它是 **20K 小时人类视频得以直接使用的前提条件**。三代之间「动作表征」与「数据来源」是耦合演进的：

| | 人类视频如何进入训练 | 代价 |
|---|---|---|
| N1 | VQ-VAE 潜动作，当成独立 "LAPA" 本体 | 潜空间与真实动作空间有 gap |
| N1.5 | FLARE 未来表征对齐 | 不需动作标签，但仍是间接监督 |
| N1.7 | **统一相对 EEF 直接作为动作标签** | 需要从视频估计手部位姿（EgoScale 的"action-labeled"即此意） |

### 7.4 一个反直觉的观察：DiT 层数 16 → 32 → 16

N1.6 把动作头翻倍到 32 层，N1.7 又砍回 16 层，同时把 action horizon 从 16 提到 40、state/action 维度从 29 扩到 132。

**合理推测**（官方未直接解释）：N1.6 的 32 层是在**弱骨干 + 需要 adapter** 的条件下靠动作头容量补偿；N1.7 骨干换成推理更强的 Cosmos-Reason2 后，语义负担回到 System 2，System 1 只需负责运动生成，因此可以瘦身，把预算换成**更长的动作块（40 步）与更宽的动作维度（132）**——后者显然是为全身控制（SONIC，含腿+臂+手）与多机身统一接口准备的。

---

## 8. 严谨数学推导

### 8.1 Flow Matching 动作生成（四代共用）

**记号**：$t$ 为环境时间步，$\tau \in [0,1]$ 为 flow-matching 时间（去噪进度），$H$ 为动作块长度。

动作块定义为

$$A_t = [a_t,\ a_{t+1},\ \dots,\ a_{t+H-1}]$$

给定真值动作块 $A_t$、采样噪声 $\epsilon \sim \mathcal{N}(0, I)$ 与 flow 时间 $\tau$，**加噪动作块**沿真值与噪声之间的直线插值构造：

$$A_t^{\tau} = \tau A_t + (1-\tau)\,\epsilon$$

注意此处约定 $\tau=1$ 为**干净动作**、$\tau=0$ 为**纯噪声**（与部分文献相反，读代码时需留意）。

对该条件路径求关于 $\tau$ 的导数，得到需要被学习的**目标速度场**：

$$\frac{\mathrm{d}A_t^{\tau}}{\mathrm{d}\tau} = A_t - \epsilon$$

GR00T 令网络 $V_\theta$ 预测其**相反数** $\epsilon - A_t$（即从干净指向噪声的方向），训练损失为

$$\mathcal{L}_{\text{fm}}(\theta) = \mathbb{E}_{\tau,\ \epsilon,\ (A_t, \varphi_t, q_t)}\Big[\big\|V_\theta(\varphi_t,\ A_t^{\tau},\ q_t) - (\epsilon - A_t)\big\|^2\Big]$$

其中 $\varphi_t$ 是 VLM 输出的视觉-语言 token，$q_t$ 是本体感觉状态嵌入。

**时间步采样分布**（沿用 π₀ 的设计，偏向低 $\tau$ 即高噪声区域）：

$$p(\tau) = \mathrm{Beta}\!\left(\frac{s-\tau}{s};\ 1.5,\ 1\right), \qquad s = 0.999$$

**推理（$K$ 步 forward Euler）**：先采样 $A_t^0 \sim \mathcal{N}(0,I)$，随后迭代

$$A_t^{\tau + 1/K} = A_t^{\tau} + \frac{1}{K}\,V_\theta(\varphi_t,\ A_t^{\tau},\ q_t)$$

实践中 $K = 4$ 在所有本体上均可用。注意这里的更新是 $+\frac{1}{K}V_\theta$，配合 $V_\theta \approx \epsilon - A_t$ 的符号约定与 $\tau$ 从 0 增到 1 的方向，整体自洽。

更完整的 flow matching 理论推导见 [[wiki/analyses/Flow Matching 完整数学推导]]。

### 8.2 潜动作（Latent Action, VQ-VAE）

设视频帧对 $(x_t,\ x_{t+H})$，编码器 $E_\phi$、解码器 $D_\psi$、码本 $\mathcal{C} = \{e_1,\dots,e_M\}$。

连续潜动作与其量化：

$$z_t = E_\phi(x_t,\ x_{t+H}), \qquad \hat{z}_t = \arg\min_{e \in \mathcal{C}} \|z_t - e\|_2$$

VQ-VAE 训练目标（标准三项：重建 + 码本 + commitment，$\mathrm{sg}[\cdot]$ 为 stop-gradient）：

$$\mathcal{L}_{\text{VQ}} = \underbrace{\big\|D_\psi(\hat{z}_t,\ x_t) - x_{t+H}\big\|^2}_{\text{重建}} + \underbrace{\big\|\mathrm{sg}[z_t] - \hat{z}_t\big\|^2}_{\text{码本}} + \beta\underbrace{\big\|z_t - \mathrm{sg}[\hat{z}_t]\big\|^2}_{\text{commitment}}$$

**关键设计**：训练完成后，GR00T **只取编码器**当作逆动力学模型，并且使用**量化前的连续嵌入** $z_t$（而非 $\hat{z}_t$）作为动作标签，把它当作一个名为 "LAPA" 的虚拟本体，套用 §8.1 的同一个 $\mathcal{L}_{\text{fm}}$ 训练：

$$\mathcal{L}_{\text{LAPA}} = \mathbb{E}\Big[\big\|V_\theta(\varphi_t,\ z_t^{\tau},\ q_t) - (\epsilon - z_t)\big\|^2\Big], \qquad z_t^\tau = \tau z_t + (1-\tau)\epsilon$$

用连续嵌入而非量化码字，是为了保留动作幅度的连续信息（码本只有 $M$ 个离散方向）。

### 8.3 FLARE：未来潜表征对齐（N1.5 引入）

FLARE 在 flow matching 之外增加一项**表征层面**的对齐损失。总目标为

$$\mathcal{L} = \mathcal{L}_{\text{fm}} + \lambda_{\text{FLARE}}\,\mathcal{L}_{\text{align}}, \qquad \lambda_{\text{FLARE}} = 0.2$$

其思想是：不去**生成**未来帧 $x_{t+k}$（那需要一个昂贵且易被像素细节干扰的视频生成目标），而是让策略内部的某组 token 去**预测未来观测经目标编码器得到的嵌入**：

$$\mathcal{L}_{\text{align}} = \mathcal{D}\Big(h_\theta(\varphi_t,\ q_t),\ \ \mathrm{sg}\big[f_{\text{target}}(x_{t+k})\big]\Big)$$

其中 $h_\theta$ 是从策略中引出的预测头，$f_{\text{target}}$ 是（通常冻结或 EMA 的）目标编码器，$\mathcal{D}$ 为距离度量。

**为什么这解锁了人类视频**：$\mathcal{L}_{\text{align}}$ **完全不需要动作标签**——只需要 $(x_t,\ x_{t+k})$ 帧对。因此任何视频（包括人类第一视角视频）都能直接贡献梯度。这正是 N1.5 能"从人类视频学操作新物体"（0-shot 15.0%、后训练 55.0%）的机制来源。

> ⚠️ 上式为依据 N1.5 官方博客描述与 FLARE 项目定位重建的**一般形式**；$h_\theta$ 的具体接入位置、$f_{\text{target}}$ 的选择与 $\mathcal{D}$ 的具体形式需查阅 FLARE 原文核实，本页尚未据原文校验。

### 8.4 相对动作空间（N1.6 引入，N1.7 统一到 EEF）

设 $s_t$ 为当前状态（关节角或末端位姿）。

**绝对动作空间**（N1 / N1.5）直接回归目标：

$$a_k^{\text{abs}} = s_{t+k}, \qquad k = 0,\dots,H-1$$

**相对动作空间**（N1.6+）回归相对**当前**状态的增量：

$$a_k^{\text{rel}} = s_{t+k} \ominus s_t$$

其中 $\ominus$ 对平移分量为普通减法，对旋转分量为 $SE(3)$ 上的相对位姿。若末端位姿 $T_t \in SE(3)$，则

$$\Delta T_k = T_t^{-1}\,T_{t+k}$$

执行时由 $\hat{T}_{t+k} = T_t\,\Delta \hat{T}_k$ 还原绝对目标。

**为什么相对表征有利于泛化**：$a^{\text{rel}}$ 的分布**与场景中物体的绝对位置无关**，只与"接下来怎么动"有关，因此分布更集中、跨场景更可迁移。**为什么小数据下会误差累积**（N1.6 明确的副作用）：每步都以估计的 $T_t$ 为参考系，参考系误差会随执行链传播；而绝对动作每步都锚定到全局系，具有自纠正性。

**N1.7 的跨本体统一**：设人类本体的手部位姿为 $T^{\text{human}}_t$、机器人末端位姿为 $T^{\text{robot}}_t$。两者的绝对空间完全不可比，但增量

$$\Delta T_k = (T_\cdot)_t^{-1}\,(T_\cdot)_{t+k}$$

对二者都是 $SE(3)$ 中的同一类量。因此**同一组动作标签、同一个 Action Decoder 头**可同时接收人类视频与机器人遥操作数据——这就是 20K 小时 EgoScale 数据得以直接进入预训练的数学基础。

---

## 9. 系列的开放问题

1. **长程组合任务仍是硬伤**。N1.6 官方自述"多任务语言跟随与 OOD 泛化仍具挑战"；N1.7 的 SimplerEnv 尾部任务（`put_eggplant_in_sink` 2.0%、`place_in_closed_drawer` 7.0%）都是需要"先开容器再放入"的两阶段任务，佐证了这一点。平均分掩盖了结构性短板。
2. **缺少统一的跨代对照表**。N1.5 给了完整的 vs N1 对照，但 N1.6 只给定性结论，N1.7 只给自身 benchmark。**"N1.7 与 N1.6 性能相当"是官方口径，没有公开逐任务数据支撑。**
3. **相对动作在小数据下的误差累积**尚无系统解法（N1.6 只是提示注意）。
4. **RL 后训练缺位**。相较 π 系列的 [[wiki/concepts/rl/RECAP]] 路线，GR00T 四代都停留在模仿学习 + 数据规模化，官方管线里没有 RL 微调环节。可对照 [[wiki/comparisons/VLA RL 微调方法对比]]。
5. **EgoScale scaling law 的外推边界未知**。log-linear 关系在 1K→20K 小时区间成立，但何时饱和、以及"验证损失→真机成功率"的相关性在分布外任务上是否保持，均未验证。

---

## 10. 实践建议（若要上手 N1.7）

- **动作空间**：优先按官方推荐配置 relative EEF；但如果你的数据集只有几十条 demo，注意 N1.6 提出的误差累积风险，可考虑先用绝对动作做基线对照。
- **归一化统计量**：任务分布接近预训练数据 → 用预训练统计量；差异大 → 用后训练统计量（否则欠拟合）。
- **`state_dropout_prob`**：默认偏高（CLI 0.2 / model config 0.8）。强依赖本体感觉的任务应调低。
- **过拟合**：N1.6/N1.7 收敛快，需要更强的状态正则 + 数据增强 + 与预训练数据协同训练。
- **准入门槛**：`nvidia/Cosmos-Reason2-2B` 是 **gated 仓库**，必须先在 HF 申请权限并 `huggingface-cli login`，否则加载基座 checkpoint 会直接 `GatedRepoError`。
- **评测方差**：图像增强非确定性带来 5–6% 跑间方差，单次结果不足以下结论。
- **实时性**：异步 rollout 建议开启 RTC；追求吞吐用 TensorRT（H100 上 11.7 Hz → 35.9 Hz）。

---

## 参考来源

- GR00T N1 论文：arXiv [2503.14734](https://arxiv.org/abs/2503.14734)（v1 2025-03-18，v2 2025-03-27）
- [GR00T N1.5 官方博客](https://research.nvidia.com/labs/gear/gr00t-n1_5/)（2025-06-11）
- [GR00T N1.6 官方博客](https://research.nvidia.com/labs/gear/gr00t-n1_6/)（2025-12-15）
- [NVIDIA/Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)（N1.7 GA）
- EgoScale：arXiv [2602.16710](https://arxiv.org/abs/2602.16710)
- [HuggingFace N1.7 博客](https://huggingface.co/blog/nvidia/gr00t-n1-7) ⚠️ 部分技术细节与仓库 config 冲突，见 §5.1
