---
type: source
tags: [VLA, FlowMatching, KnowledgeInsulation, ActionExpert, StopGradient, PhysicalIntelligence]
sources: [raw/assets/papers/VLA/Driess 等 - 2025 - Knowledge Insulating Vision-Language-Action Models Train Fast, Run Fast, Generalize Better.pdf]
created: 2026-07-03
updated: 2026-07-03
---

# Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better

**arXiv**: 2505.23705
**作者**: Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z. Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, Sergey Levine（Physical Intelligence）
**发表日期**: 2025-05-29
**HuggingFace**: https://hf.co/papers/2505.23705
**项目主页**: https://pi.website/research/knowledge_insulation
**摄取日期**: 2026-07-03
**摄取来源**: 用户本地 PDF + MinerU 转换

[[raw/assets/papers/VLA/Driess 等 - 2025 - Knowledge Insulating Vision-Language-Action Models Train Fast, Run Fast, Generalize Better.pdf]]
[[raw/sources/papers/VLA/Driess 等 - 2025 - Knowledge Insulating Vision-Language-Action Models Train Fast, Run Fast, Generalize Better/Driess 等 - 2025 - Knowledge Insulating Vision-Language-Action Models Train Fast, Run Fast, Generalize Better.md]]

---

## 一句话摘要

给 VLM 接上"从零初始化"的连续动作专家（flow matching action expert）会因梯度回传污染预训练权重、拖慢收敛并损害语言跟随；本文提出 **Knowledge Insulation（知识绝缘）**——用**离散动作 token（FAST）做表示学习信号训练 backbone**，同时训练连续动作专家但**切断其到 backbone 的梯度流（stop-gradient）**，再辅以 **VLM 数据 co-training**，从而做到"训练快、推理快、泛化更好"。

---

## 核心背景与动机

### 问题：给 VLM 加连续动作专家会破坏预训练知识

VLA 的核心承诺是把 web-scale VLM 的语义知识迁移到机器人控制。但实时控制需要**连续、高频、精确**的动作输出，而 VLM 原生是**离散 token 自回归**架构。业界两条路都各有硬伤：

1. **自回归离散 VLA（如 π₀-FAST）**：把连续动作离散化后自回归解码。
   - 推理慢：预测 1 秒动作块在 RTX4090 上 ≈750 ms，控制频率 <2 Hz（π₀-FAST ≈1.3 Hz），高频灵巧任务不可用。
   - 离散化有损，精细/动态动作能力下降。

2. **连续动作专家 VLA（如 π₀）**：加一个 diffusion / flow matching 的 action expert，推理可达 10 Hz。
   - **但新增的 action expert 是从零初始化的**，训练时其梯度回传会"污染" backbone。
   - 论文明确指出：naive 地加 action expert **显著损害语言跟随能力**、且**训练收敛慢**（π₀ 需要约 7.5× 于本文方法的训练步数才能达到相近性能）。

### 三个失败模式（Sec. 4）

- **自回归 VLA 太慢**：序列解码 + 离散分辨率受限。
- **机器人专用适配模块吃不到 VLM 预训练红利**：action expert 从零初始化，其梯度干扰预训练权重 → 语言跟随退化（gradient interference）。
- **冻结 backbone 也不行**：VLM 预训练没有机器人数据，冻结后的表征不足以支撑高性能策略——实验里冻结方案直接 0% 性能。

### 关键洞察

> 只要 backbone 被**离散动作**这一独立学习信号同时训练，就可以**安全地切断 action expert 的梯度**——因为 transformer 各层激活里已经包含足够推断动作的信息，action expert 只需通过 attention 从 backbone "读取"这些信息即可。离散信号负责塑造表征，连续专家负责精确快速地生成——两者解耦，互不干扰。

![[raw/sources/papers/VLA/Driess 等 - 2025 - Knowledge Insulating Vision-Language-Action Models Train Fast, Run Fast, Generalize Better/images/92c3b5806d25d9694c477310496343753cc3aff5be42b6ebf408a3f122d283ad.jpg]]

---

## 方法：Knowledge Insulation

建立在 [[wiki/entities/models/π₀.₅]] / π₀ 架构之上（PaliGemma backbone + 小 action expert，MoE 式分权重 transformer）。三个组件叠加：

### 组件 1：离散/连续动作联合训练（joint-training）

模型输出空间 $y = (a_{1:H},\, y^{\ell,a})$，其中 $a_{1:H}$ 是连续动作块，$y^{\ell,a}$ 同时包含语言 token 与 **FAST 离散动作 token**。联合损失：

$$
\mathcal{L}_{\mathrm{CO\text{-}VLA}}(\theta) = \mathbb{E}_{\mathcal{D},\tau,\omega} \Big[ -\sum_{j=1}^{n-1} M_j^{\ell}\log p_\theta(\hat{\ell}_{j+1}\mid x_{1:j}) + \alpha\, M^{\mathrm{act}} \big\| \omega - a_{1:H} - f_\theta^a(a_{1:H}^{\tau,\omega}) \big\|^2 \Big]
$$

- 第一项：语言 + 离散 FAST 动作 token 的交叉熵（next-token prediction），作为 **backbone 的表示学习目标**，让训练快速稳定收敛。
- 第二项：连续动作块的 flow matching 损失（$\pi_0$ 风格），噪声动作 $a_{1:H}^{\tau,\omega}=\tau a_{1:H}+(1-\tau)\omega,\ \omega\sim\mathcal N(0,\mathbf I)$，预测向量场 $\omega-a_{1:H}$。
- $\alpha$ 为损失系数；$M^\ell$ 是语言 loss mask，$M^{\mathrm{act}}$ 是动作 mask indicator，可灵活混合不同模态数据。
- **关键 attention 约束**：离散 FAST 动作 token 与连续动作 token **互不 attend**，避免两种动作表示间信息泄漏。

推理时**只用 action expert** 走几步 flow 积分生成连续动作 → 快且精确。离散动作分支只在训练时充当表示学习信号。

### 组件 2：VLM 数据 co-training

在动作数据之外混入通用 VLM 数据（image captioning、VQA、bounding box 定位）与机器人 planning 数据。作用是**减少灾难性遗忘**、保住 VLM 语义知识，尤其对新物体的语义泛化（OOD follow rate）至关重要。

### 组件 3：知识绝缘 = stop-gradient（核心贡献）

Backbone 与 action expert 仅通过 attention 交互。把 attention 的 softmax 概率分块：

$$
P = \operatorname{softmax}\big(Q(X)K(X)^T + A\big) = \begin{pmatrix} P_{bb} & 0 \\ P_{ab} & P_{aa} \end{pmatrix}
$$

其中 $P_{bb}$ 是 backbone→backbone，$P_{ab}$ 是 action expert→backbone，$P_{aa}$ 是 action expert→action expert。为切断 action expert 到 backbone 的梯度，在计算里插入 stop-gradient 算子 $\mathrm{sg}(\cdot)$：

$$
\begin{pmatrix} P_{bb} & 0 \\ P_{ab} & P_{aa} \end{pmatrix} = \operatorname{softmax}\left(\begin{pmatrix} Q_b(X_b)K_b(X_b)^T & 0 \\ Q_a(X_a)\,\mathrm{sg}\big(K_b(X_b)^T\big) & Q_a(X_a)K_a(X_a)^T \end{pmatrix} + A\right)
$$

value 聚合同样加 stop-gradient：

$$
E = \begin{pmatrix} E_b \\ E_a \end{pmatrix} = \begin{pmatrix} P_{bb}V_b(X_b) \\ P_{ab}\,\mathrm{sg}\big(V_b(X_b)\big) + P_{aa}V_a(X_a) \end{pmatrix}
$$

**效果**：信息**单向**从 VLM 流向 action expert，没有任何 VLM 嵌入 attend 到 action expert，也没有 action expert 的梯度污染 backbone。附带好处——由于 flow 损失现在只作用于独立的一组权重，可以直接设 $\alpha=1$，无需再调损失权重。

### 架构与训练细节（附录 B）

- Backbone：PaliGemma 2B（width=2048, depth=18, mlp_dim=16384, heads=18, kv_heads=1, head_dim=256）。
- Action expert：更小的 transformer（width=1024, mlp_dim=4096），约 **300M 参数**；动作 horizon $H=50$。
- 时间步 $\tau$ 采样沿用 π₀ 的偏低时间步 Beta 分布：$p(\tau)=\mathrm{Beta}(\frac{s-\tau}{s};\alpha{=}1.5,\beta{=}1),\ s{=}0.999$。
- τ 经 sinusoidal 编码 + MLP 后用 adaptive RMSNorm 注入到 action expert 每一层。
- Attention：图像/语言/text-state 用 full prefix mask；FAST 动作 token attend prefix 并自回归；action expert attend prefix 与自身，但**不 attend FAST 动作 token**。

### 状态表示（附录 C）

三种 proprioceptive state 表示：**text state**（离散化后当普通文本，token 数多）、**special token state**（离散 bin 绑定特殊 token，s 个 token）、**continuous state**（affine 投影直接进 embedding）。实验表明本方法对 text state 与 continuous state 都工作良好；special token 最差。

---

## 实验结果

评测涵盖真实世界灵巧长程操作（table bussing / shirt folding / items in drawer / 4 项移动双臂任务）+ 仿真 LIBERO + 真实 DROID，多机器人本体。

- **整体性能**：真实任务上本方法一致最优。"items in drawer"（held-out 环境，需语言跟随 + 精确开抽屉）中所有 baseline 明显更差，joint-training（无 stop-gradient）与 π₀ 语言跟随差，π₀-FAST 慢且开抽屉不精确，HybridVLA（允许 AR token attend 连续动作）显著掉点。
- **DROID**：本方法 $0.55\pm0.09$ > π₀ $0.49\pm0.09$ > π₀-FAST $0.45\pm0.09$。
- **LIBERO**（Tab.1）：**LIBERO-90 与 LIBERO-Spatial 达到 SOTA**。
  - Ours(from generalist): Spatial 98.0 / Object 97.8 / Goal 95.6 / Long 85.8 / 90 → 96.0
  - Ours(from scratch): 96.6 / 97.2 / 94.6 / 84.8 / 92.7
  - 唯 LIBERO-10(Long) 略逊于 MoDE(94.0) 与 OpenVLA-OFT(94.5)。
- **收敛速度**（Fig.6b）：本方法训练与 π₀-FAST 一样快；**π₀（纯 flow）需约 7.5× 训练步数**才达相近性能。
- **语言跟随**（Fig.4b）：stop-gradient 显著改善语言跟随；若加 VLM co-training，即使无 stop-gradient 的 joint-training 也能获得不错的语言跟随。Transfusion 语言跟随好于带 action expert 的 π₀（因为它复用 backbone 权重、唯一从零初始化的是动作投影）→ 佐证"从零初始化机器人适配器的梯度会破坏预训练权重"这一假设。
- **VLM→机器人语义迁移**（Fig.7, 移动操作新物体 OOD）：VLM 数据 co-training 对未见物体泛化尤为关键。
- **离散表示消融**：用 FAST 做表示学习优于 naive tokenization；naive 版仍好于纯连续动作，但更弱；naive 下按 stride=5 子采样 token 优于稠密 naive。

---

## 局限性与未来工作

- 同时训练连续 + 离散输出使**训练计算量约增加 20%**；但因收敛更快，wall-clock 上仍显著快于纯 diffusion 的 π₀。
- 语言跟随虽有改善但**远未完美**——训练数据中的相关性仍会让模型有时忽略语言指令。

---

## 与已有方法的关系

- **[[wiki/entities/models/π₀.₅]]**：π₀.₅ 首次用"先 FAST 预训练、后加随机初始化 action expert 联合训练"的两阶段做法；本文把它**形式化并扩展为单阶段配方**（backbone 用离散 token 适配、action expert 同时训连续动作），并系统消融 stop-gradient 与 co-training。Knowledge Insulation 即 π₀.₅ 中同名技术的正式出处。
- **π₀（arXiv 2410.24164）**：本文的连续 action expert 架构基础；同时是被诊断"训练慢 + 语言跟随差"的主要对照。
- **π₀-FAST（arXiv 2501.09747）**：提供 FAST（DCT + 量化 + BPE）离散动作 tokenizer，本文借它当表示学习信号；对照其自回归推理慢的缺陷。
- **Transfusion / HybridVLA**：同一 backbone 内 denoise 连续输入 / AR+diffusion 联合；本文与 HybridVLA 最相似，但关键区别是**离散动作 token 不允许 attend 连续动作**且切断梯度。
- **[[wiki/concepts/generative-models/Flow Matching]]**：action expert 的连续动作建模基础。
- **[[wiki/concepts/vla/Vision-Language-Action 模型]]**：本页所属大类。

---

## 新概念追踪

**首次出现，追踪中（源自本页，达到 ≥2 源可升级为 concept 页）**:
- **Knowledge Insulation（知识绝缘）**：离散动作表示学习信号 + action expert stop-gradient 的组合配方；已在 π₀.₅ 与本文出现 → **接近 ≥2 源阈值，建议升级为 concept 页**。
- **Gradient Interference from Randomly-Initialized Adapters**：从零初始化机器人适配器的梯度破坏预训练 VLM 权重的现象与诊断。
- **Discrete-token-as-representation-learning-signal**：离散动作 token 仅在训练期充当 backbone 表征塑造信号、推理期弃用。

---

## 关联页面

- [[wiki/entities/models/π₀.₅]] — Knowledge Insulation 的应用与前身
- [[wiki/concepts/vla/Vision-Language-Action 模型]] — VLA 总览
- [[wiki/concepts/generative-models/Flow Matching]] — action expert 的连续动作生成基础
- [[wiki/comparisons/VLA RL 微调方法对比]] — VLA 训练配方对比入口
