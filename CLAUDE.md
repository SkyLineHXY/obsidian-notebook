
# Wiki Schema — Harness Engineering

**系统指令**：这是当前工作区的核心配置文件。每次会话开始时，你必须首先阅读此文件，随后阅读 `wiki/index.md`，然后再执行用户的具体请求。

## 1. 目录结构与权限控制

| 层级目录        | 维护者    | 读写权限            | 描述                               |
| :---------- | :----- | :-------------- | :------------------------------- |
| `raw/`      | 用户     | **Claude 仅只读**  | 原始资料层。用户负责收集，你绝不能修改其中的任何文件。      |
| `wiki/`     | Claude | **Claude 全权读写** | 知识库层。你负责根据下述规则创建、更新和维护此目录。用户仅阅读。 |
| `CLAUDE.md` | 共同     | 共同演进            | 本配置说明书。                          |

```text
Harness Engineering/
├── CLAUDE.md                 ← 本文件：Schema 配置
├── raw/                      ← 原始资料层
│   ├── README.md
│   ├── assets/               ← 非 Markdown 原始文件（PDF、DOCX、PPTX 等二进制格式）
│   │   ├── papers/           ← 学术论文原文（PDF 为主）
│   │   │   └── <topic>/      ← 主题子目录，例如 Generative Model/、VLA+RL/、IL(Imitation Learning)/
│   │   ├── figure/           ← 图片、截图等视觉资产
│   │   └── ppt/              ← 演示文稿原文
│   └── sources/              ← Markdown 格式数据来源（可直接被 Claude 阅读）
│       ├── papers/           ← 由 raw/assets/papers/ 中的 PDF 经 MinerU 转换而来（保留原路径结构）
│       │   └── <topic>/      ← 与 assets/papers/ 镜像的主题子目录
│       ├── blogs/            ← 网页博客、专栏文章、新闻等（Markdown 格式）
│       ├── github/           ← GitHub 仓库 README / 项目说明（Markdown 格式）
│       └── others/           ← 其他零散碎片资料
├── wiki/                     ← 知识库层
│   ├── index.md              ← 全局内容索引（每次 Ingest 后强制更新）
│   ├── log.md                ← 系统操作日志（仅追加）
│   ├── overview.md           ← 知识库顶层综合综述
│   ├── entities/             ← 实体页：人物、组织、产品、工具等
│   │   ├── models/           ← 模型实体（π₀.₅, SmolVLA, π₀.₇ 等）
│   │   ├── frameworks/       ← 框架实体（RLinf, StarVLA, ARIS 等）
│   │   ├── tools/            ← 工具实体（Claude Code 等）
│   │   ├── hardware/         ← 硬件/系统实体（Franka, UMI, RoboTwin, Ubuntu 等）
│   │   ├── systems/          ← 系统方案实体（HIL-SERL 等）
│   │   └── people/           ← 人物实体
│   ├── concepts/             ← 概念页：核心原理、方法论、技术术语
│   │   ├── rl/               ← 强化学习（Offline RL, RECAP, DPPO 等）
│   │   ├── imitation-learning/ ← 模仿学习（HG-DAgger, ACT 等）
│   │   ├── generative-models/ ← 生成模型（DDPM, Flow Matching, Diffusion Policy）
│   │   ├── vla/              ← VLA 模型（Vision-Language-Action 模型）
│   │   ├── benchmarks/       ← 基准评测（LIBERO 等）
│   │   └── infrastructure/   ← 基础设施（PREEMPT_RT 实时内核等）
│   ├── sources/              ← 来源摘要页：每篇原始资料的提炼与总结
│   │   ├── agent-systems/    ← Agent 系统 / 研究 harness（ARIS 等）
│   │   ├── rl-finetuning/    ← RL 微调表达性策略（ReinFlow, DPPO, πRL 等）
│   │   ├── generative/       ← 生成模型（DDPM, Flow Matching）
│   │   ├── imitation-learning/ ← 模仿学习（Diffusion Policy, ACT 等）
│   │   ├── vla/              ← VLA 模型（VLASH, π₀.₇ 等）
│   │   ├── frameworks/       ← 框架与基础设施（LeRobot, RLinf, StarVLA）
│   │   ├── infrastructure/   ← 系统安装与配置（Ubuntu, 实时内核, 控制器规范）
│   │   └── guides-tools/     ← 指南与工具（Embodied-AI-Guide, MinerU）
│   ├── comparisons/          ← 对比分析页：跨来源的横向综合对比（保持扁平）
│   └── analyses/             ← 深度分析页：复杂问题解答、专题探索（保持扁平）
```

## 2. 页面格式与约定

### 2.1 Frontmatter (YAML)

新建或更新的每一个 wiki 页面，都必须包含以下格式的 YAML 头部：


```yaml
---
type: entity | concept | source | comparison | analysis
tags: [tag1, tag2]
sources: [来源文件名或 URL]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### 2.2 命名与存储规范

| **页面类型**       | **存储路径格式**                                           | **内容说明**          |
| -------------- | ------------------------------------------------------ | ----------------- |
| **Entity**     | `wiki/entities/<category>/实体名.md`                     | 具体的人、组织、产品、工具。    |
| **Concept**    | `wiki/concepts/<category>/概念名.md`                     | 核心原理、方法论、技术术语。    |
| **Source**     | `wiki/sources/<theme>/YYYY-MM-DD 标题.md`               | 对单篇原始资料的摘要、要点提炼。  |
| **Comparison** | `wiki/comparisons/对比主题.md`                            | 跨多个来源的横向对比与综合分析。  |
| **Analysis**   | `wiki/analyses/分析主题.md`                               | 深度分析、复杂问题解答、专题探索。 |

> 新增页面时根据主题选择合适的子目录；若现有子目录都不合适，可新建。

**Source 页命名扩展**：
- 论文：`<theme>/YYYY-MM-DD <主题关键词>.md`（如 `rl-finetuning/2026-04-19 DPPO.md`）
- 博客：`<theme>/YYYY-MM-DD <原文标题>.md`
- GitHub 仓库：`<theme>/YYYY-MM-DD <owner-repo 或仓库名>.md`（如 `frameworks/2026-04-19 LeRobot.md`）
- `YYYY-MM-DD` 以**摄取日期**为准，而非原始资料的发表日期。

### 2.3 交叉引用规则 (Obsidian Wiki-link)

1. 页面间引用必须使用双括号：`[[页面名]]`。
    
2. 文本中提到的所有核心实体/概念，都应尝试链接到对应的维基页。若对应页面不存在，需评估是否满足创建条件（见 2.5）。
    
3. **关键约束**：在 `wiki/sources/` 下的每个摘要页顶部，**必须**使用干净的 wiki-link 链接到对应的 `raw` 文件（例如：`[[raw/sources/Effective harnesses.md]]`）。**严禁**使用反引号包住链接，确保图谱视图可识别。
    
4. 图片引用规范：原始资料中的图片保存在 `raw/assets/`，在 wiki 中引用格式为：`![[raw/assets/图片名.png]]`。读取包含图片的 Markdown 时，先处理文本，再单独读取相关图像获取视觉上下文。
    

### 2.4 数学公式规范（强制 LaTeX）

**核心规则**：wiki 中所有数学表达式**必须**使用 LaTeX 语法，由 Obsidian 的 MathJax 引擎渲染。

| 场景    | 格式              | 示例                         |
| ----- | --------------- | -------------------------- |
| 行内公式  | `$...$`         | `策略梯度为 $\nabla J(\theta)$` |
| 独立公式块 | `$$...$$`（独占段落） | 复杂方程、推导链                   |

**禁止的写法（Lint 时必须修复）**：

| ❌ 错误                               | ✅ 正确                                    |
| ---------------------------------- | --------------------------------------- |
| 代码块包裹公式 `` `ak+1 = ak + vθ·Δtk` `` | `$a_{k+1} = a_k + v_\theta \Delta t_k$` |
| Unicode 数学符号 `Σ ln N(...)`         | `$\sum_k \ln \mathcal{N}(\ldots)$`      |
| 纯文本下标 `theta'`、`sigma^2`           | `$\theta'$`、`$\sigma^2$`                |
| 纯文本分布符号 `N(0,I)`                   | `$\mathcal{N}(0, I)$`                   |

> **Ingest 时的执行要求**：在将原始文章中的公式转录到 wiki 页面时，若原文使用伪代码、纯文本或 Unicode 表示数学，必须主动将其转换为规范 LaTeX 格式，不得原样复制。

---

### 2.5 页面创建时机（Karpathy 自下而上原则）

**核心思想**：概念必须从多篇来源中自然浮现，严禁对单一文章进行过度抽象建页。

- **Source 页**：无条件创建。每摄取一篇原始资料，必须创建一个对应的来源摘要页。
- **Entity / Concept / Comparison 页**：必须满足**≥2篇不同来源**的阈值才能创建。

### 2.6 数学与理论强制推导机制
**核心思想**：知其然，更要知其所以然。知识库拒绝停留在“科普层面”，必须深入到底层数学。
触发条件：当提取/创建属于以下领域的 Concept 或 Analysis 页面时：
- 强化学习 (RL)、模仿学习 (IL)、控制理论。
- 生成式 AI (扩散模型、自回归模型、VLA 等)。
- 核心统计学、优化算法、微积分原理。

执行要求（必须包含独立的 `## 严谨数学推导` 章节）：
1. 符号定义：在推导前，必须先列出所有用到的数学变量及其物理/算法含义（如 $s_t$: state, $a_t$: action, $\pi_\theta$: policy）。
2. 起点与终点：明确说明推导的“假设前提/优化目标（Objective Function）”以及最终的“结论公式”。
3. 逐步推导 (Step-by-Step)：不得跳跃步骤。
4. 物理意义解释：在推导的每个关键等号后面，用文字简述该步转换的数学依据或物理直觉。
5. 严格遵守 2.4 节的 LaTeX 规范。

**执行逻辑**：
1. **首次出现**：将概念记录在来源摘要页的 `tags` 或 `Related Concepts` 中，**不要**创建独立的维基页。
2. **再次出现**：主动回溯已有知识库，发现第 2 次出现时**立即创建**独立的 Concept/Entity 页并更新交叉引用。

---

## 3. 核心工作流 (Workflows)
### 3.1 摄取 (Ingest)
**触发条件**：用户指令包含"摄取"、"处理"、"Ingest"等。
**操作步骤**：
1. **预处理（文档解析）**：若目标文件为 `.pdf`、`.docx` 或 `.pptx` 格式，**必须强制调用 `mineru` skill**，将文件转换为 Markdown（详见 §5）。
   - **输出路径规则**：若源文件位于 `raw/assets/papers/<topic>/`，则转换后的 `.md` 文件必须保存到 `raw/sources/papers/<topic>/`，保持相同的主题子目录路径，文件名与 PDF 同名（后缀改为 `.md`）。
   - 转换完成后，使用 `Read` 工具读取生成的 Markdown 文件。对超过 100 行的 `.md` 来源文件，分段读取（每次 `Read` 指定 `offset` + `limit`）。
   - **严禁**用 `Read` 工具直接读取 PDF 二进制流；**严禁**调用 `mineru-document-explorer` skill（该 skill 存在 bug）。
2. **阅读**：仔细阅读转换后的 markdown 内容及相关图片。
3. **讨论**：与用户互动，确认关键要点和提炼侧重点。
4. **推导检查**：如果是算法/模型论文，主动询问用户是否需要在摘要页或相关 Concept 页补充“核心定理的完整证明与推导”（遵循 2.6 节）。
5. **创建**：建立 wiki/sources/YYYY-MM-DD 标题.md 摘要页。
6. **评估**：运用“创建时机原则”，若概念达到 ≥2 篇来源的阈值，则创建新的实体页/概念页。
7. **更新**：更新相关的已有实体页、概念页和 wiki/index.md。
8. **记录日志**：在 wiki/log.md 尾部追加记录。

### 3.2 查询 (Query)

触发条件：用户针对知识库内容提问。

**操作步骤**：

1. **检索** 优先读取 `wiki/index.md` 定位相关页面，随后深入读取具体页面及关联链接。
2. **综合** 生成详实回答，并必须明确标注信息来源的具体维基页面。
3. **归档** 若回答产生了有价值的交叉对比或新见解，主动将其写入 `wiki/comparisons/` 作为新页面。
4. **记录日志** 在 `wiki/log.md` 尾部追加记录。
### 3.3 审查 (Lint)

触发条件：用户指令包含"检查 wiki"、"Lint"等。

**操作步骤**：你需要全面扫描 `wiki/` 目录，检查并逐一修复以下问题：

- **事实冲突**：不同页面对同一事实的描述是否矛盾。
- **孤儿页面**：是否存在没有任何入站链接的页面。
- **无效链接**：是否有被双括号引用但实际不存在的文件（评估是否需要新建或修改链接）。
- **信息陈旧**：是否有明显被新资料推翻但未及时更新的内容。
- **索引错位**：`index.md` 是否与实际物理文件完全同步。
- **聚合建议**：主动建议是否有值得新建的汇总页或对比页。
- **公式格式违规**：扫描是否存在以代码块、Unicode 数学符号或纯文本书写的数学公式（见 2.4），若发现则原地替换为规范 LaTeX。
- 最后，在日志中记录修复了多少个问题。
---

## 4. 索引与日志规范

### 4.1 Index 维护规则 (`wiki/index.md`)

每次 Ingest 或新建页面后，必须更新对应的表格行，格式严格如下：

`| [[wiki/sources/<theme>/2026-04-18 标题]] | 一句话简介 | 2026-04-18 |`

### 4.2 Log 维护规则 (`wiki/log.md`)

日志为 Append-only（仅追加）。每次操作结束后写入，格式严格如下：


```Markdown
## [YYYY-MM-DD] 操作类型 | 标题/简述
- 动作细节 1
- 动作细节 2
```

_操作类型限定为：`init`, `ingest`, `query`, `lint`, `update`。_

---

## 5. 文档解析规范（MinerU Skill）

**强制要求**：摄取任何 PDF、DOCX、PPTX 文件时，**必须**先调用 `mineru` skill 将其转换为 Markdown，再用 `Read` 工具读取。**严禁**调用 `mineru-document-explorer` skill（该 skill 存在 bug，不可用）。

### 5.1 PDF / DOCX / PPTX 标准解析流程

```
# Step 1 — 调用 mineru skill，将文件转换为 Markdown
# skill 会写一个 wrapper 脚本到临时目录并执行，token 从 MINERU_TOKEN 环境变量读取

# Step 2 — 确定输出路径（针对 raw/assets/papers/ 下的论文）
# 源文件：raw/assets/papers/<topic>/paper.pdf
# 输出目录：raw/sources/papers/<topic>/
# 最终产物：raw/sources/papers/<topic>/paper/paper.md
#   → 读取时直接使用该路径
Read("D:/Desktop/Obsidian文件/个人笔记/raw/sources/papers/<topic>/paper/paper.md")

# Step 3 — 大型 Markdown（> 100 行）分段读取
Read("...paper.md", offset=0, limit=200)
Read("...paper.md", offset=200, limit=200)
# ...依此类推
```

### 5.1.1 路径镜像规则（`assets → sources`）

| 原始 PDF 路径                             | 转换后 Markdown 输出目录             |
| ------------------------------------- | ----------------------------- |
| `raw/assets/papers/<topic>/paper.pdf` | `raw/sources/papers/<topic>/` |

实际产物路径（mineru_v2.py 在 output 目录下创建与文件同名的子目录）：
`raw/sources/papers/<topic>/<stem>/<stem>.md`

### 5.2 执行要点

- **Token**：`MINERU_TOKEN` 已保存在 `~/.claude/settings.json` 的 `env` 节，无需手动传入。
- **Python 命令**：Windows 上使用 `python`，**不要用 `python3`**（该命令被系统钩子拦截）。
- **不传 `--token` 参数**：token 通过环境变量传递，命令行中的 JWT 字符串会触发安全钩子。
- **wrapper 脚本模式**：通过写临时文件再执行的方式调用脚本，避免在 Bash 命令中暴露 token。

### 5.3 决策速查

| 场景                | ✅ 正确操作                       | ❌ 禁止操作                                      |
| ----------------- | ---------------------------- | ------------------------------------------- |
| PDF / DOCX / PPTX | `mineru` skill → `Read` 读 MD | `Read` 直读 PDF；调用 `mineru-document-explorer` |
| `raw/assets/papers/<topic>/` 下的论文 PDF | 输出目录指定为 `raw/sources/papers/<topic>/` | 输出到任意其他位置 |
| Markdown 超 100 行  | `Read` 分段（offset + limit）    | 一次性整体读取                                     |
| 图片理解              | `Read`（读取图片文件）               | 跳过图片内容                                      |

### 5.4 Lint 时的额外检查项

执行 §3.3 Lint 时，额外检查：
- 确认近期新增的 PDF 来源是否经过 `mineru` skill 转换（而非 Read 直读 PDF）。
- 若 wiki/sources 页中存在公式明显缺失或结构混乱，提示可能是解析不完整，建议重新运行 `mineru` skill。

---

> **Schema v1.9** — 2026-05-17 | 如需调整约定，直接修改本文件，Claude 将在下次解析时严格遵循新规则。