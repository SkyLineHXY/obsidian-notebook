# Wiki Schema — Harness Engineering

**系统指令**：每次会话首先阅读本文件，再读 `wiki/index.md`，然后再执行用户请求。

## 1. 目录结构与权限

| 层级 | 维护者 | Claude 权限 | 说明 |
|---|---|---|---|
| `raw/` | 用户 | **只读** | 原始资料。`assets/` 存二进制（PDF/DOCX/PPTX/图片），`sources/` 存 Markdown（含 `papers/<topic>/<stem>/<stem>.md` + `images/<hash>.jpg`） |
| `wiki/` | Claude | **读写** | 知识库。`index.md` 全局索引、`log.md` 操作日志、`overview.md` 顶层综述、`entities/`、`concepts/`、`sources/`、`comparisons/`、`analyses/` |
| `CLAUDE.md` | 共同 | 共同演进 | 本配置 |

子目录约定（保持现有划分；新增按主题就近放置，必要时新建）：
- `entities/`：`models/`、`frameworks/`、`tools/`、`hardware/`、`systems/`、`people/`
- `concepts/`：`rl/`、`imitation-learning/`、`generative-models/`、`vla/`、`benchmarks/`、`infrastructure/`
- `sources/`：`agent-systems/`、`rl-finetuning/`、`generative/`、`imitation-learning/`、`vla/`、`vla-rl/`、`frameworks/`、`data-collection/`、`data-efficiency/`、`infrastructure/`、`guides-tools/`、`lab-automation/`
- `comparisons/` 与 `analyses/` 保持扁平。

## 2. 页面规范

### 2.1 Frontmatter（所有 wiki 页面必含）

```yaml
---
type: entity | concept | source | comparison | analysis
tags: [Tag1, Tag2]
sources: [文件路径或 URL]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### 2.2 命名与存储

| 类型 | 路径格式 | 说明 |
|---|---|---|
| Entity | `wiki/entities/<category>/实体名.md` | 人/组织/产品/工具 |
| Concept | `wiki/concepts/<category>/概念名.md` | 原理/方法论/术语 |
| Source | `wiki/sources/<theme>/YYYY-MM-DD 标题.md` | 单篇资料摘要；`YYYY-MM-DD` 为**摄取日期** |
| Comparison | `wiki/comparisons/对比主题.md` | 跨来源横向对比 |
| Analysis | `wiki/analyses/分析主题.md` | 深度专题 |

### 2.3 链接 & 图片

- 页面引用统一双括号：`[[wiki/.../页面]]`；提到核心实体/概念时尽量链接。
- **Source 页必含**干净 wiki-link 回链 raw 文件（**不得**用反引号包住）：例如 `[[raw/sources/papers/<topic>/<stem>/<stem>.md]]`。
- **论文图片嵌入**（强制）：MinerU 输出的图片位于 `raw/sources/papers/<topic>/<stem>/images/<hash>.jpg`，原始 MD 用相对路径 `![](images/<hash>.jpg)`。**写入 wiki 时必须改写为 Obsidian wiki-link 形式**：
  ```markdown
  ![[raw/sources/papers/<topic>/<stem>/images/<hash>.jpg]]
  *Figure：descriptive caption from paper*
  ```
  每个 source 页嵌入 1–3 张关键图（taxonomy/architecture/key-result 优先）。**不要复制图片**——直接引用 raw 路径。

### 2.4 数学公式（强制 LaTeX）

行内 `$...$`，块级 `$$...$$`（独占段落）。**禁止**：代码块包裹公式、Unicode 数学符号、纯文本 `theta'`/`N(0,I)`。

| ❌ 错误 | ✅ 正确 |
|---|---|
| `` `ε_θ`、`ᾱ_t`、`N(0,I)` `` | `$\varepsilon_\theta$`、`$\bar\alpha_t$`、`$\mathcal{N}(0, I)$` |
| 算法伪代码 fenced block 内含 Unicode 公式 | 改写为 Markdown 列表 + 行内 LaTeX |

Ingest 时若原文用伪代码/Unicode/纯文本写数学，**必须**主动转为规范 LaTeX。

### 2.5 创建时机（Karpathy 自下而上）

- **Source 页**：无条件每篇创建。
- **Entity / Concept / Comparison**：≥ 2 篇不同来源的阈值才能创建。首次出现仅记 tag/related；第二次出现立即建页并回填交叉引用。

### 2.6 数学/理论强制推导

对 RL / IL / 控制 / 生成式 AI / 优化等领域的 Concept 或 Analysis 页，**必须**含 `## 严谨数学推导` 节，包含：符号定义、假设与目标、逐步推导、关键步骤的物理/算法直觉，全部使用 LaTeX。

## 3. 工作流

### 3.1 Ingest（"摄取/处理/Ingest"）

1. **解析**：PDF/DOCX/PPTX **必须**先调用 `mineru` skill（见 §5），输出 `raw/sources/papers/<topic>/<stem>/<stem>.md` + `images/`。**禁用** `mineru-document-explorer` skill（有 bug）。
2. **读取**：`Read` 转换后的 MD（> 100 行分段 offset+limit）；嵌入图片需先 `Read` 图片获取视觉上下文。
3. **讨论**：与用户确认关键要点。
4. **数学推导确认**：算法/模型论文主动询问是否需要补充完整证明（§2.6）。
5. **选图**：识别 1–3 张关键图（taxonomy/architecture/key-result），记录 `<hash>.jpg`。
6. **写 Source 页**：含 frontmatter、`[[raw/...]]` 回链、`![[raw/.../images/<hash>.jpg]]` 嵌入 + caption。
7. **评估升级**：达 ≥ 2 篇来源的概念/实体立即建独立页。
8. **更新**：相关已有页 + `wiki/index.md`。
9. **记录**：`wiki/log.md` 追加条目（§4.2）。
10. **同步**：执行 §6 Git 同步。

### 3.2 Query（用户提问）

1. 先读 `wiki/index.md` 定位，再深入相关页面与链接。
2. 综合作答，**标注信息来源具体页面**。
3. 若产生有价值的交叉对比/新见解 → 写入 `wiki/comparisons/` 或 `wiki/analyses/`，并 §6 同步。
4. `wiki/log.md` 追加。

### 3.3 Lint（"检查 wiki / Lint"）

全面扫描并修复：事实冲突、孤儿页面、无效双括号链接、信息陈旧、index.md 与文件系统失同步、值得新建的聚合/对比页、公式格式违规（代码块/Unicode/纯文本数学 → LaTeX）、近期 PDF 是否经 mineru 转换、解析不完整迹象（建议重跑 mineru）。最后日志记录修复数，§6 同步。

## 4. 索引 & 日志

### 4.1 `wiki/index.md`

每次 Ingest/新建后更新对应表格行：

```
| [[wiki/sources/<theme>/YYYY-MM-DD 标题]] | 一句话简介 | YYYY-MM-DD |
```

### 4.2 `wiki/log.md`（Append-only）

```markdown
## [YYYY-MM-DD] 操作类型 | 简述
- 动作细节
```

操作类型：`init` / `ingest` / `query` / `lint` / `update`。

## 5. MinerU 解析速查

**强制**：PDF/DOCX/PPTX 必须先 `mineru` skill → 再 `Read`。**禁用** `mineru-document-explorer`。

| 项 | 规则 |
|---|---|
| 路径镜像 | `raw/assets/papers/<topic>/<file>.pdf` → 输出目录 `raw/sources/papers/<topic>/`；产物 `<topic>/<stem>/<stem>.md` + `<stem>/images/<hash>.jpg` |
| Token | `MINERU_TOKEN` 已在 `~/.claude/settings.json` `env` 节；**不要**传 `--token`（会触发安全 hook） |
| 调用方式 | 写 wrapper 脚本到临时目录再 `python` 执行（避免命令行暴露 token） |
| Python | Windows 用 `python`（**不要** `python3`，被 hook 拦截） |
| MD 读取 | > 100 行用 `Read offset=+limit` 分段 |
| 图片 | MD 内 `![](images/<hash>.jpg)`（相对）→ 嵌入 wiki 时改写为 `![[raw/sources/papers/<topic>/<stem>/images/<hash>.jpg]]`（绝对 wiki-link） |
| Lint | 检查近期 PDF 来源是否经 mineru 转换；公式缺失/结构混乱提示重跑 |

## 6. GitHub 同步

仓库：`https://github.com/SkyLineHXY/obsidian-notebook.git`（branch: `main`）。

**触发**：Ingest 完成、Lint 修复、Query 归档、CLAUDE.md 更新后**强制**同步。

**命令**：

```bash
git add wiki/ CLAUDE.md raw/sources/
git commit -m "docs(wiki): <一句话描述>"
git -c http.proxy="" -c https.proxy="" push origin main
```

**代理绕过**：本机有 SOCKS 代理（`127.0.0.1:8080`），push 必须加 `-c http.proxy="" -c https.proxy=""` 清空代理。

Commit message 例：`docs(wiki): ingest DPPO — 添加 RL 微调摘要页及扩散策略概念页`。

---

> **Schema v2.1** — 2026-05-23 | 重构精简版（新增 §2.3 图片嵌入规则）。
