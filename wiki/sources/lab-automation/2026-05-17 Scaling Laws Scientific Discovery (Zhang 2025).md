---
type: source
tags: [Lab Automation, Autonomous Scientist, Scaling Laws, Robot Learning, Scientific Discovery]
sources: [raw/assets/papers/Lab Automation/Zhang - 2025 - Scaling Laws Scientific Discovery.pdf]
created: 2026-05-17
updated: 2026-05-17
---

# Scaling Laws in Scientific Discovery with AI and Robot Scientists

**arXiv**: 2503.22444
**作者**: Pengsong Zhang, Heng Zhang, Huazhe Xu, Renjun Xu, Zhenting Wang, Cong Wang, Animesh Garg, Zhibin Li, et al.
**发表日期**: 2025-03-28
**HuggingFace**: https://hf.co/papers/2503.22444
**摄取日期**: 2026-05-17
**摄取来源**: HuggingFace paper_search（agent 自动检索）

[[raw/assets/papers/Lab Automation/Zhang - 2025 - Scaling Laws Scientific Discovery.pdf]]
[[raw/sources/papers/Lab Automation/Zhang - 2025 - Scaling Laws Scientific Discovery/Zhang - 2025 - Scaling Laws Scientific Discovery.md]]

---

## 一句话摘要

该论文论证了 AI 与机器人科学家（Robot Scientist）结合后科学发现能力的规模化效应，提出自主通才科学家（Autonomous Generalist Scientist）可通过整合文献综述、假设生成、实验执行与论文写作实现全流程科研自动化。

---

## 核心背景与动机

### 问题：科学发现的瓶颈在执行层

LLM 已展现出强大的文献综述和假设生成能力，但自主科学发现的瓶颈在于：**如何将高层科学假设转化为物理实验操作**，并在实验结果反馈下迭代修正。纯软件 AI scientist 无法执行需要物理接触的实验步骤。

### 关键洞察

类比 LLM 训练的 scaling law：**AI + 机器人结合的科学发现能力**也呈现规模化效应——随着机器人执行能力（操作任务覆盖率）、AI 推理能力（模型规模）和实验数据量的增长，科学发现的吞吐量和质量持续提升。

---

## 核心框架：Autonomous Generalist Scientist

### 全流程科研自动化

```
文献检索 + 综述
    ↓
假设生成（LLM）
    ↓
实验设计
    ↓
物理实验执行（Robot Scientist）← 核心新贡献
    ↓
数据分析 + 结果解读
    ↓
论文写作
```

### Robot Scientist 的角色

Robot Scientist 是具备通用操作能力的机器人系统，需支持：
- **多样化实验仪器操作**：移液枪、离心机、光谱仪、显微镜等；
- **长时域任务执行**：单次实验可能包含数十步操作；
- **跨学科泛化**：从生物、化学到材料科学的实验操作。

### 规模化法则的类比

设科学发现质量 $Q$ 与三个维度的规模参数有关：

$$Q \approx f(C_{\text{robot}},\; C_{\text{AI}},\; D_{\text{exp}})$$

其中 $C_{\text{robot}}$ 为机器人执行能力覆盖率，$C_{\text{AI}}$ 为 AI 模型规模（参数量或推理 FLOPs），$D_{\text{exp}}$ 为实验数据积累量。作者论证三者均呈现收益递增效应。

---

## 与已有方法的关系

- **[[wiki/sources/lab-automation/2026-05-17 Intelligent Science Laboratory Position (Zhang 2025)]]**：两篇均为 position/vision 论文，本文侧重规模化法则视角，前者侧重认知-具身整合架构；
- **[[wiki/sources/lab-automation/2026-05-17 ChemBot (Huang 2026)]]**、**[[wiki/sources/lab-automation/2026-05-17 BioMARS (Qiu 2025)]]**：均为本文"Robot Scientist"愿景的具体落地系统；
- **[[wiki/sources/agent-systems/2026-05-16 ARIS]]**：ARIS 是纯软件的研究 harness，本文则强调物理实验执行层的必要性，两者形成互补。

---

## 新概念追踪

**首次出现，追踪中**：
- **Autonomous Generalist Scientist**：覆盖科研全流程（文献→假设→实验→论文）的自主 AI+机器人系统；仅来源 42
- **Scaling Laws for Scientific Discovery**：AI+机器人结合的科学发现能力随计算、机器人覆盖率、数据量规模化增长的法则；仅来源 42
- **Robot Scientist**：具备通用实验室操作能力的机器人科学家；仅来源 42（与 BioMARS/ChemBot 的具体系统形成实例关联）

---

## 关联页面

- [[wiki/sources/lab-automation/2026-05-17 Intelligent Science Laboratory Position (Zhang 2025)]] — 互补的认知-具身整合 position paper
- [[wiki/sources/lab-automation/2026-05-17 ChemBot (Huang 2026)]] — Robot Scientist 愿景在化学实验室的落地
- [[wiki/sources/lab-automation/2026-05-17 BioMARS (Qiu 2025)]] — Robot Scientist 愿景在生物实验室的落地
- [[wiki/sources/agent-systems/2026-05-16 ARIS]] — 软件层的研究 harness，与本文构成软件+物理的互补视角
