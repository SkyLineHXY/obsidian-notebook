---
type: comparison
tags: [VLA, AgenticFramework, LongHorizon, InferenceTime, SAP, StateGap, Comparison]
sources: [raw/assets/papers/VLA/Yang 等 - 2025 - Agentic Robot A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents.pdf, raw/assets/papers/VLA/Pang 等 - 2026 - Sci-VLA Agentic VLA Inference Plugin for Long-Horizon Tasks in Scientific Experiments.pdf]
created: 2026-06-02
updated: 2026-06-02
---

# Agentic VLA 系统对比：Agentic Robot vs Sci-VLA

两篇论文共享同一个**元思想**：用 **LLM/VLM agent 在推理期编排 VLA**，把长程操作的可靠性问题从"重训一个更强的 base VLA"转移到"在 VLA 外围加一层 agentic 控制环"。两者都**不重训 base VLA**、都聚焦 LIBERO-Long 式/科学实验式的**长程操作**，但攻击的失效模式与解法正交，恰好互补。

- [[wiki/sources/agent-systems/2026-06-02 Agentic Robot (Yang 2025)]]（来源 52）
- [[wiki/sources/agent-systems/2026-06-02 Sci-VLA (Pang 2026)]]（来源 53）

---

## 一、核心失效模式：各打一个靶

| 维度 | **Agentic Robot** | **Sci-VLA** |
|------|-------------------|-------------|
| 目标失效模式 | 子目标执行**失败 / 误差累积**（抓取失败、卡死不自知） | 原子任务**衔接处 State Gap**（任务间缺过渡 → OOD 卡死抖动） |
| 失效定位 | 任务**内**（执行不可靠） | 任务**间**（过渡缺失） |
| 隐喻 | "执行后没有人验收 → 错误级联" | "两段动作之间断了一截 → 接不上" |

---

## 二、机制对比：闭环验证 vs 过渡插桩

| 维度 | **Agentic Robot（SAP 闭环）** | **Sci-VLA（推理插件）** |
|------|------------------------------|--------------------------|
| 编排协议 | **SAP**：感知-规划-执行-验证四阶段异步有限状态机 | **两模块插件**：过渡动作生成 + 过渡动作插入 |
| 高层规划 | LRM Planner（GPT-4o）分解子目标，受 Atomic Skill Library 约束 | 无显式 planner，输入即给定原子任务序列 |
| Agent 介入时机 | **每 20 帧周期性**巡检（主动、常态） | **仅任务切换处**介入（被动、事件触发） |
| 核心干预 | **VLM 验证器内省** → 继续/重试/恢复 | **LLM 生成过渡动作代码** → 断开-插入-重连 VLA |
| 验证/纠错 | 两阶段验证器：完成判定 + 卡死诊断 + 恢复动作 $\pi_{\text{rec}}$ | 无验证；过渡动作内置安全约束（避障、释爪） |
| Agent 输出形态 | Yes/No 判定 + 离散恢复动作 | **可执行机器人代码**（`gripper(open)`、`move(...)`、`adjust(target_pose)`） |
| 是否需适配训练 | 验证器需 LoRA 微调（~500 三元组） | **完全训练无关**（纯 prompt + 检索） |

---

## 三、组件与实现

| 项 | **Agentic Robot** | **Sci-VLA** |
|----|-------------------|-------------|
| Planner | GPT-4o（LRM） | GPT-5.2（仅做语义检索 + 代码生成） |
| Executor (base VLA) | **OpenVLA**（自回归） | $\pi_0$ / $\pi_{0.5}$（扩散/flow）+ $\pi_0$-FAST（自回归） |
| Verifier / Agent | Qwen2.5-VL-3B-Instruct（LoRA 微调） | GPT-5.2（零样本，模板约束） |
| 频率 | Executor 10 Hz / Verifier 0.5 Hz | 事件驱动（每原子任务设最大运行时 $T$） |
| 观测 | 第三人称 + 腕部双视角滑动缓冲 | 当前主相机视图 + 当前/目标关节位姿 |

---

## 四、评测与场景

| 项 | **Agentic Robot** | **Sci-VLA** |
|----|-------------------|-------------|
| 场景 | 通用桌面操作（LIBERO） | 科学实验室（Autobio 数字孪生 + 真机） |
| 基准 | LIBERO 4 子集，平均 **79.6%**（SOTA） | "Cleaning Table" + 6 科学长程任务（3/5/8 步） |
| 关键数字 | LIBERO-Long 61.6%，超 SpatialVLA +6.1%、OpenVLA +7.4% | 平均每原子任务成功率 **+42%**（裸 VLA 后续任务 0% → 可用） |
| Sim-to-Real | 仅仿真验证，真机为 future work | 真机热循环仪任务验证迁移 |

---

## 五、综合判断

- **互补而非竞争**：理论上可叠加——Sci-VLA 的过渡插桩负责"接上断点"，Agentic Robot 的验证-恢复环负责"接上后执行不出错"。前者解决 OOD 衔接，后者解决执行内省。
- **训练成本**：Sci-VLA 更极致（零训练，纯推理）；Agentic Robot 需为验证器做轻量 LoRA（~500 样本）。
- **可解释性**：两者都显著优于端到端 VLA——Agentic Robot 暴露子目标 + Yes/No 验证轨迹，Sci-VLA 暴露可读的过渡动作代码。
- **共同局限**：都不提升 base VLA 的**原子动作精度上限**（Sci-VLA 明言高精度子任务仍受 base VLA 制约；Agentic Robot 在对称物体精细放置上 -5%）。

---

## Related

- [[wiki/concepts/vla/Vision-Language-Action 模型]] — 两系统的 base executor 谱系
- [[wiki/concepts/benchmarks/LIBERO]] — Agentic Robot 主基准
- [[wiki/sources/vla/2026-04-23 VLASH]] — 第三种推理期增强路线（异步时延，不重训）
- [[wiki/sources/agent-systems/2026-05-16 ARIS]] — agentic harness 的"研究流程"对照版
