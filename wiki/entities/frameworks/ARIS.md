---
type: entity
tags: [ARIS, Harness, Autonomous-Research, Agent-Systems, Claude-Code, Cross-Model-Review]
sources: [https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep, arXiv:2605.03042]
created: 2026-05-16
updated: 2026-05-16
---

# ARIS (Auto-Research-In-Sleep)

**ARIS ⚔️** — Lightweight, Markdown-only autonomous ML research harness. Orchestrates cross-model collaboration (executor + external reviewer) across the full research lifecycle: idea discovery → experiment automation → paper writing → rebuttal.

> *"A methodology, not a platform. What matters is the research workflow — take it wherever you go."*

---

## Identity

| Field | Value |
|---|---|
| **Full name** | Auto-Research-In-Sleep |
| **Acronym** | ARIS ⚔️🌙 |
| **Current version** | v0.4.7 (2026-05-16) |
| **Paper** | arXiv:2605.03042 |
| **Repo** | [wanshuiyin/Auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) |
| **Primary executor** | [[wiki/entities/tools/Claude Code]] (also: Codex CLI, Cursor, Trae, Antigravity, Windsurf) |
| **Primary reviewer** | GPT-5.5 via Codex MCP (also: Kimi, GLM, MiniMax, DeepSeek, Gemini) |
| **License** | Open source |

---

## Three-Layer Architecture

```
┌──────────────────────────────────┐
│  Assurance Layer (3-stage audit) │  experiment-audit → result-to-claim → paper-claim-audit
├──────────────────────────────────┤
│  Orchestration Layer (5 WFs)     │  W1 → W1.5 → W2 → W3 → W4 + Meta
├──────────────────────────────────┤
│  Execution Layer (65+ skills)    │  SKILL.md files, plain Markdown, no framework
└──────────────────────────────────┘
```

- **Execution**: 65+ `SKILL.md` files, each a plain Markdown prompt callable by any LLM agent
- **Orchestration**: 5 named Workflows that chain skills into end-to-end research pipelines
- **Assurance**: 3-stage audit cascade ensuring evidence-to-claim integrity before submission

---

## Core Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| **W1**: `/idea-discovery` | New research direction | Survey → Brainstorm → Novelty check → Pilot → Rank |
| **W1.5**: `/experiment-bridge` | Have a plan | Implement → GPT code review → Deploy → Collect |
| **W2**: `/auto-review-loop` | Have results | 4-round review/fix cycle, overnight score lifting |
| **W3**: `/paper-writing` | Ready to write | Narrative → Outline → Figures → LaTeX → PDF |
| **W4**: `/rebuttal` | Reviews arrived | Parse → Strategy → Draft → Safety gates |
| **M**: `/meta-optimize` | Anytime | Analyze usage logs → Propose skill improvements |

---

## Cross-Model Review Design

**Core insight**: executor reviewing its own work creates self-play blind spots. ARIS separates:
- **Executor** (Claude Code / Codex): fast, fluid, writes code and prose
- **Reviewer** (GPT-5.5 xhigh via Codex MCP): slower, deliberate, fresh context per round

The reviewer never sees fix summaries from prior rounds (Reviewer Independence Protocol), preventing sycophantic convergence. Two-model design chosen because 1→2 provides the biggest gain; n>2 has diminishing returns with higher cost.

---

## Key Design Principles

1. **Heterogeneous models** — executor ≠ reviewer model family, minimizing shared blind spots
2. **Modular skills** — each skill is a standalone `SKILL.md`, zero coupling
3. **Composability** — skills chain into workflows; workflows chain into full pipelines
4. **Portability** — works on Claude Code, Codex CLI, Cursor, Trae, Antigravity, Windsurf
5. **Persistent memory** — Research Wiki accumulates papers/ideas/experiments across sessions; meta-optimize reads usage logs to improve the harness itself

---

## Effort Levels

| Level | Token multiplier | Use case |
|---|---|---|
| `lite` | 0.4× | Quick exploration |
| `balanced` | 1× (default) | Normal research |
| `max` | 2.5× | Pre-submission hardening |
| `beast` | 5–8× | Top-venue sprint, all knobs to max |

---

## Related Pages

- [[wiki/sources/agent-systems/2026-05-16 ARIS]] — source summary page
- [[wiki/entities/tools/Claude Code]] — primary executor platform
