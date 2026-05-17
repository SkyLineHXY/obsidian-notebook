---
type: source
tags: [VLA, Bayesian, Information Collapse, Language Grounding, OOD Generalization, SimplerEnv, RoboCasa, Latent Action Queries, PMI]
sources: [https://hf.co/papers/2601.15197]
created: 2026-05-17
updated: 2026-05-17
---

# BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries

**arXiv**: 2601.15197  
**作者**: Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang 等  
**发表日期**: 2026-01-21  
**HuggingFace**: https://hf.co/papers/2601.15197（54 upvotes）  
**摄取日期**: 2026-05-17  
**摄取来源**: HuggingFace Daily Papers（agent 自动检索）

---

## 一句话摘要

通过贝叶斯分解将 VLA 拆分为视觉先验 $p(a \mid v)$ 与语言条件后验 $\pi(a \mid v, \ell)$ 双分支，以最大化 Conditional PMI 惩罚视觉捷径，无需新数据即在 OOD SimplerEnv 上提升 11.3%。

---

## 核心背景与动机

### 问题：Information Collapse

现有 VLA 训练存在一个被忽视的结构性缺陷：**目标驱动的数据采集方式**导致数据集中语言指令几乎可以从视觉观测直接预测。

$$I(a; \ell \mid v) \approx 0$$

当指令与视觉高度相关时，语言对动作的条件互信息趋近于零。模型因此退化为**视觉-only 策略**（vision-only policy），在推理时实际上忽略语言约束 —— 在 in-distribution 任务中表现尚可，但在 OOD 设置（新指令、新任务组合）下严重失效。

### 关键洞察

问题根源不在于模型容量或数据量，而在于训练目标**没有显式惩罚视觉捷径**。只要预测损失可以通过视觉路径最小化，模型就没有动力真正理解语言。

---

## 方法：BayesianVLA

### 框架设计

引入可学习的 **Latent Action Queries**，构建双分支架构：

| 分支 | 形式 | 输入 | 功能 |
|---|---|---|---|
| Vision-only Prior | $p(a \mid v)$ | 视觉观测 $v$ | 捕捉与语言无关的动作先验 |
| Language-conditioned Posterior | $\pi(a \mid v, \ell)$ | 视觉 + 语言 $v, \ell$ | 条件化语言指令的完整策略 |

Latent Action Queries 作为可学习 token 嵌入两个分支的解码器，共享视觉特征但在是否注入语言 token 上分离。

### 训练目标：最大化 Conditional PMI

$$\text{PMI}(a; \ell \mid v) = \log \frac{\pi(a \mid v, \ell)}{p(a \mid v)}$$

最终损失：

$$\mathcal{L}_{\text{BayesianVLA}} = \mathcal{L}_{\text{BC}} - \lambda \cdot \mathbb{E}\left[\log \frac{\pi(a \mid v, \ell)}{p(a \mid v)}\right]$$

其中 $\mathcal{L}_{\text{BC}}$ 为标准行为克隆损失，$\lambda$ 为 PMI 正则化权重。

**直觉**：后验与先验的比值越大，说明动作对语言的依赖越强 —— 惩罚"语言不重要"的预测，奖励显式解释语言指令的动作。

### 核心优势

- **无需新数据**：在已有数据集上改变训练目标，无需额外采集
- **无需架构大改**：在现有 VLA 骨干上插入 Latent Action Queries 即可
- **理论保证**：PMI 最大化等价于最小化策略对语言的条件独立性（信息论视角）

---

## 实验结果

### 基准测试

| 基准 | 设置 | 提升 |
|---|---|---|
| **SimplerEnv（OOD）** | 包含未见指令的开放世界评测 | **+11.3%** |
| **RoboCasa** | 多任务机器人厨房场景 | substantial gains（具体数值见论文） |

### 消融分析（推测自摘要）

- 去除 PMI 损失 → 退化为 vision-only 策略，OOD 性能大幅下降
- 去除 Latent Action Queries → 双分支解耦失效

---

## 与已有方法的关系

本文的 Information Collapse 概念与以下方向互补：

- **VLA 架构设计**（[[wiki/concepts/vla/Vision-Language-Action 模型]]）：现有 VLA 主要关注多模态融合，鲜少从信息论角度审视语言-动作耦合
- **Diffusion Policy**（[[wiki/concepts/generative-models/Diffusion Policy]]）：行为克隆范式的代表；BayesianVLA 的 $\mathcal{L}_{\text{BC}}$ 项可基于任意策略架构（包括 diffusion）
- **π₀.₇**（[[wiki/sources/vla/2026-04-29 π₀.₇]]）：通过 episode metadata 解歧义混合质量数据；BayesianVLA 则从语言-动作互信息角度解歧义

---

## 新概念追踪

**首次出现，追踪中**：
- **Information Collapse**：VLA 训练中因目标驱动数据采集导致语言-动作条件互信息趋零的现象；仅本来源系统命名与分析
- **Latent Action Queries**：用于构建 vision-only 与 language-conditioned 双分支的可学习 token；仅本来源提出
- **Conditional PMI for VLA**：将 PMI 最大化用作 VLA 语言基础训练目标；仅本来源

---

## 关联页面

- [[wiki/concepts/vla/Vision-Language-Action 模型]] — 所属模型类别
- [[wiki/concepts/generative-models/Diffusion Policy]] — 典型 BC 策略架构
- [[wiki/sources/vla/2026-04-23 VLASH]] — 同为 VLA 泛化改进方向
- [[wiki/sources/vla/2026-04-29 π₀.₇]] — 同为 VLA OOD / 泛化能力研究
