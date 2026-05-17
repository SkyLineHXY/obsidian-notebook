---
type: source
tags: [Tactile Learning, Imitation Learning, Fine-Grained Manipulation, Lab Automation, Visuo-Tactile]
sources: [raw/assets/papers/Lab Automation/Zhu - 2025 - Touch in the Wild.pdf]
created: 2026-05-17
updated: 2026-05-17
---

# Touch in the Wild: Learning Fine-Grained Manipulation with a Portable Visuo-Tactile Gripper

**arXiv**: 2507.15062
**作者**: Xinyue Zhu, Binghao Huang, Yunzhu Li
**发表日期**: 2025-07-20
**HuggingFace**: https://hf.co/papers/2507.15062
**摄取日期**: 2026-05-17
**摄取来源**: HuggingFace paper_search（agent 自动检索）

[[raw/assets/papers/Lab Automation/Zhu - 2025 - Touch in the Wild.pdf]]
[[raw/sources/papers/Lab Automation/Zhu - 2025 - Touch in the Wild/Zhu - 2025 - Touch in the Wild.md]]

---

## 一句话摘要

该工作提出了一款集成触觉传感器的便携式末端执行器，结合跨模态表示学习框架，使机器人能够通过模仿学习掌握包括移液管液体转移在内的精密实验室操作任务。

---

## 核心背景与动机

### 问题：精密操作中视觉信息的局限

在实验室操作（如移液、试管插入）中，接触力和触觉反馈是决定操作成功的关键信号——仅凭视觉无法感知液体重量变化、接触面压力或滑动趋势。传统触觉传感器体积大、不便携，难以推广到野外（in-the-wild）数据采集场景。

### 关键洞察

设计**便携式视触觉夹爪（Portable Visuo-Tactile Gripper）**，使触觉传感器与视觉传感器紧密集成，并通过**跨模态对比表示学习**将视觉与触觉信号映射到统一语义空间，让策略在测试时即使缺少触觉也能泛化。

---

## 方法：跨模态视触觉学习框架

### 末端执行器设计

- 集成 GelSight 类触觉传感器（高分辨率接触成像）；
- 轻量便携，可挂载于标准机械臂末端；
- 支持"野外"数据采集（in-the-wild collection），降低实验室外部署门槛。

### 跨模态表示学习

设模态对齐损失为：

$$\mathcal{L}_{\text{align}} = -\mathbb{E}_{(v, t) \sim \mathcal{D}} \left[ \log \frac{\exp(\text{sim}(f_v(v),\, f_t(t)) / \tau)}{\sum_{j} \exp(\text{sim}(f_v(v),\, f_t(t_j)) / \tau)} \right]$$

其中 $v$ 为视觉观测，$t$ 为对应的触觉读数，$f_v, f_t$ 分别为视觉和触觉编码器，$\tau$ 为温度系数，$\text{sim}(\cdot)$ 为余弦相似度。

通过对比学习对齐后，视觉表示隐式包含触觉语义，策略在推理时无需触觉传感器亦可运行。

### 策略训练（模仿学习）

在采集的视触觉演示数据上训练行为克隆（BC）策略：

$$\pi^* = \arg\min_\pi \mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{demo}}} \left[ \mathcal{L}({\pi}(s), a) \right]$$

其中 $s$ 包含视觉 + 触觉融合表示，$a$ 为专家动作。

---

## 实验结果

**任务覆盖（含实验室场景）**：
- **试管插入（Test Tube Insertion）**：精密对孔定位，成功率超出视觉-only 基线；
- **移液管液体转移（Pipette-based Fluid Transfer）**：需感知液体重量，触觉反馈关键；
- 其他通用精密操作任务。

**核心发现**：
- 跨模态表示学习使策略在测试时仅用视觉即可达到接近视触觉联合输入的性能；
- 便携设计使数据采集效率显著提升，支持 in-the-wild 泛化。

---

## 与已有方法的关系

- **[[wiki/concepts/imitation-learning/ACT]]**：同为基于模仿学习的精密操作策略框架，Touch in the Wild 扩展了感知模态（触觉）；
- **[[wiki/sources/lab-automation/2026-05-17 ChemBot (Huang 2026)]]**：ChemBot 解决化学实验的长时域规划，本文解决实验室精密操作的感知局限，两者互补；
- **[[wiki/sources/data-collection/2026-04-24 UMI (Chi 2024)]]**：同为便携式数据采集装置，UMI 基于 GoPro 鱼眼视觉，本文扩展为视触觉联合采集。

---

## 新概念追踪

**首次出现，追踪中**：
- **Portable Visuo-Tactile Gripper**：集成触觉传感器的便携末端执行器，支持野外数据采集；仅来源 40
- **Cross-Modal Representation Learning（视触觉）**：通过对比学习对齐视觉与触觉嵌入，使策略在推理时视觉可替代触觉；仅来源 40
- **In-the-Wild Tactile Data Collection**：野外环境下的触觉演示数据采集范式；仅来源 40

---

## 关联页面

- [[wiki/concepts/imitation-learning/ACT]] — 精密操作 IL 策略
- [[wiki/sources/data-collection/2026-04-24 UMI (Chi 2024)]] — 便携式数据采集装置对比
- [[wiki/sources/lab-automation/2026-05-17 ChemBot (Huang 2026)]] — 化学实验室操作系统
- [[wiki/sources/lab-automation/2026-05-17 Intelligent Science Laboratory Position (Zhang 2025)]] — 精密操作能力是实验室自动化的核心需求之一
