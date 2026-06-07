---
type: source
tags: [VLA, AgenticFramework, LongHorizon, SAP, TemporalVerifier, OpenVLA, LIBERO, BrainInspired, ErrorRecovery]
sources: [raw/assets/papers/VLA/Yang 等 - 2025 - Agentic Robot A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents.pdf]
created: 2026-06-02
updated: 2026-06-02
---

[[raw/sources/papers/VLA/Yang 等 - 2025 - Agentic Robot A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents/Yang 等 - 2025 - Agentic Robot A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents.md]]

# Agentic Robot（Yang et al., 2025）

**标题**：Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents
**机构**：吉林大学 / Harvard / MIT / 华中科技大学 / 南方科技大学 / Lehigh / 上海交大
**arXiv**：2505.23450　**项目主页**：https://agentic-robot.github.io

---

## 核心问题：长程操作的误差累积

长程机器人操作（摆桌、装配、打包）要求**扩展推理 + 精确执行 + 鲁棒错误恢复**。现有方法落入两类，各有致命弱点：

1. **静态规划跟随（static plan-following）**：生成固定执行序列、无自适应反馈 → 早期小偏差级联放大为灾难性失败（compounding error）。
2. **端到端视觉运动策略（end-to-end visuomotor）**：观测直接映射动作、无中间推理 → 缺乏内省机制，遇到分布外状态难以恢复。

共同缺陷：**执行期间无有效验证机制**。

---

## 核心方法：SAP（Standardized Action Procedure）

受人类组织的**标准操作流程（SOP）**与脑科学（前额叶规划、运动皮层执行、感觉-运动环验证）启发，作者提出 **SAP——一种协调协议**，将"感知-推理-执行-验证"编码为结构化闭环。三大设计原则：

1. **模块化分解（Modular Decomposition）**：复杂任务拆为可验证子目标。
2. **结构化协调（Structured Coordination）**：组件交互遵循预定义工作流，而非机会式通信。
3. **自适应验证（Adaptive Verification）**：系统性检查点实现早期错误检测与恢复。

每个 SAP 周期是一个 **agentic step**，封装完整工作流：

$$\mathcal{S}_t = \left(O_t,\; t_i,\; \mathbf{a}_t,\; \hat{y}_t\right)$$

其中 $O_t = \{I_t^r, I_t^w\}$ 为第三人称视角 + 腕部视角观测，$t_i$ 为当前子目标，$\mathbf{a}_t$ 为执行动作，$\hat{y}_t \in \{\text{Yes}, \text{No}\}$ 为验证结果。SAP 由一个**异步有限状态机** $\mathcal{M}_{\text{SAP}}$ 调度：执行器以 10 Hz（$\Delta t_{\text{exec}}=0.1\text{s}$）运行，验证器以 0.5 Hz（$\Delta t_{\text{ver}}=2\text{s}$）运行。

![[raw/sources/papers/VLA/Yang 等 - 2025 - Agentic Robot A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents/images/6a1942299d14fc42cb4d3cfeea6859cde81124f324e498f84e5362b80ffb9310.jpg]]
*Figure 1：Agentic Robot 框架总览。高层任务经 LRM Planner 在 Skill Library 指导下分解为子目标 → VLA Executer 执行 7-DoF 动作 → Verifier 巡检 → 形成 Plan(1/4)→Execute(2/4)→Verify(3/4)→Completed(4/4) 的 SAP agentic 闭环。*

---

## 三大专用组件

### 1. Planner —— LRM 子目标分解

高层指令 $T$ 转为结构化子目标序列：

$$\{t_1, t_2, \dots, t_N\} = P(T, I_0)$$

- 实现：GPT-4o，处理指令 $T$ + 可选初始图像 $I_0$（视觉锚定）。
- 每个子目标 $t_i$ 来自 **Atomic Skill Library**（如 `pick up [object]` / `place [object] in/on [location]` / `turn on/off [device]`）。
- 经验上 1–2 语义单元（动词+宾语 (+位置)）、整体 2–5 步在清晰度与可执行性间最优。

### 2. VLA Executor —— 反应式视觉运动策略

子目标 + 视觉观测 → 连续低层控制：

$$\mathbf{a}_t = \pi_{\text{exec}}(t_i, I_t^r), \qquad \mathbf{a}_t \in \mathbb{R}^7$$

前 6 维为平移+旋转，第 7 维 $g_t \in \{0,1\}$ 为夹爪开合。实现：**OpenVLA**（ViT + LLM 骨干）。执行器本身无状态，错误处理完全交给 SAP 验证环。

### 3. Temporal Verifier —— 内省式子目标评估与恢复

每隔 $\Delta t_v$（默认 20 帧 / 0.5 Hz）做两阶段评估。**第一阶段**判断子目标是否完成：

$$\hat{y}_{t_v} = \pi_{\text{ver}}(\mathcal{B}_{t_v}, t_i) \to \text{Yes / No}$$

其中 $\mathcal{B}_{t_v} = \{(I_{t_v-k}^r, I_{t_v-k}^w)\}_{k=0}^{K-1}$ 为第三人称 + 腕部视角的**滑动缓冲**（典型 $K=2$，帧间隔 5），捕捉物体位移与接触变化。验证器为 **Qwen2.5-VL-3B-Instruct**，仅用约 **500 条标注三元组** $(B_t, t_i, y)$ 经 LoRA 微调。

**第二阶段**（当 $\hat{y}_{t_v}=\text{No}$）诊断是否卡死：

$$f_t = \pi_{\text{diag}}(\mathcal{B}_{t_v}) \to \text{Stuck / StillTrying}$$

若 $f_t = \text{Stuck}$，触发恢复动作（如抬升夹爪、调整腕部）：

$$\mathbf{a}_{t+1} = \pi_{\text{rec}}(f_t, O_{t+1})$$

随后重试 $t_i$；$R_{\max}$ 次恢复失败则标记任务失败并停机（防不安全行为）。20 帧验证间隔相比 10 帧仅掉 1.2% 准确率，却减少 48% 计算量。

---

## 实验（LIBERO 基准）

四子集（Spatial / Object / Goal / Long），3 随机种子、500 试次。**平均成功率 79.6%（SOTA）**：

| 方法 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| Diffusion Policy | 78.3 | **92.5** | 68.3 | 50.5 | 72.4 |
| Octo-Base (FT) | 78.9 | 85.7 | **84.6** | 51.1 | 75.1 |
| OpenVLA (FT) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| TraceVLA (FT) | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| SpatialVLA (FT) | **88.2** | 89.9 | 78.6 | 55.5 | 78.1 |
| **Agentic Robot** | 85.8 | 89.0 | 81.8 | **61.6** | **79.6** |

- **长程任务 LIBERO-Long 优势最大**：61.6%，超 SpatialVLA **+6.1%**、超 OpenVLA **+7.4%**（base executor 即 OpenVLA）。
- 子目标级对比中，相对 OpenVLA 平均 +12.1%，最大增益 Bowl-Drawer +24%、Soup-Sauce +21%、Mug-Mug +19%。
- **局限**：Moka-Moka 等对称物体精细放置任务反而 -5%（缺乏空间约束预判）。

### 消融（LIBERO-Long，Full = 61.8%）

| 移除组件 | SR (%) | Δ |
|---------|--------|---|
| No Visual Input（仅文本 Planner） | 57.4 | −4.4 |
| No Recovery Mechanism | 59.7 | −2.1 |
| No Fine-tuned VLM（零样本验证器） | 35.3 | **−26.5** |
| No Subgoal Decomposition（≈裸 OpenVLA） | 53.7 | −8.1 |
| **Full System** | **61.8** | — |

最关键的两项是**微调验证器**（零样本验证器崩到 35.3%）与**子目标分解**。验证频率分析显示长程任务对验证稀疏度敏感（50 帧间隔掉 6 个百分点），20 帧为速度-精度最优。

![[raw/sources/papers/VLA/Yang 等 - 2025 - Agentic Robot A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents/images/fcf53d8139e4ddca8c1bc5deab8e6fc46c00e85eec0a843fbc719d50996f4350.jpg]]
*Figure 3：OpenVLA（上）vs Agentic Robot（下）在 "Put the cream cheese in the bowl" 上的对比。OpenVLA 抓取失败后夹爪撞桌、任务失败；Agentic Robot 经视觉验证检测失败 → 触发恢复（Lift the gripper）→ 重试完成。*

---

## 与本知识库其他工作的关系

- **与 [[wiki/sources/agent-systems/2026-05-16 ARIS]] 同属 agent-systems**：ARIS 是研究流程的 agentic harness；Agentic Robot 是**物理操作层**的 agentic 编排——都用"规划-执行-验证"闭环 + 内省式评审消解误差累积。
- **与 [[wiki/sources/agent-systems/2026-06-02 Sci-VLA (Pang 2026)]] 互补**：两者都用 **LLM agent 在推理期编排 VLA** 解决长程任务，但解法正交——Agentic Robot 靠**子目标级闭环验证 + 恢复**（VLM 验证器内省），Sci-VLA 靠**原子任务间过渡动作插桩**（弥合 state gap）。详见 [[wiki/comparisons/Agentic VLA 系统对比]]。
- **与 [[wiki/sources/vla/2026-04-23 VLASH]] 对照**：VLASH 优化的是单 VLA 的**推理时延/异步**；Agentic Robot 优化的是**多组件任务级可靠性**，两者皆为推理期增强、不重训 base VLA。

## Related Concepts & Entities

- [[wiki/concepts/vla/Vision-Language-Action 模型]] — OpenVLA 作为 SAP 的 Executor；Agentic Robot 是 VLA 的 agentic 编排范式
- [[wiki/concepts/benchmarks/LIBERO]] — 主评测基准，长程子集（LIBERO-Long）为核心战场
- [[wiki/concepts/imitation-learning/ACT]] — 动作分块是 VLA 执行器的基础输出范式

---

## 待沉淀概念（单来源，记入 Knowledge Gaps）

- **SAP（Standardized Action Procedure）**：感知-规划-执行-验证的结构化协调协议（仅来源 52）。
- **Temporal Verifier / Introspective Assessment**：两阶段（完成判定 + 卡死诊断）VLM 内省验证器（仅来源 52）。
- **Large Reasoning Model 子目标分解**：用 GPT-4o 类 LRM 将指令拆为 Atomic Skill Library 子目标（仅来源 52）。
- **Atomic Skill Library**：标准化动作模板库，约束 Planner 输出与 Executor 兼容（仅来源 52）。
