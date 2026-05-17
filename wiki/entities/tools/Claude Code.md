---
type: entity
tags: [Claude-Code, Anthropic, CLI, Agent, MCP, Skills]
sources: [https://docs.anthropic.com/en/docs/claude-code, https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep]
created: 2026-05-16
updated: 2026-05-16
---

# Claude Code

**Claude Code** — Anthropic's agentic CLI tool that runs Claude models in a terminal environment with full filesystem, shell, and MCP (Model Context Protocol) access. The primary executor platform for the [[wiki/entities/frameworks/ARIS]] harness and the host environment for this entire Obsidian knowledge base.

---

## Identity

| Field | Value |
|---|---|
| **Maker** | Anthropic |
| **Type** | Agentic CLI / coding assistant |
| **Documentation** | [docs.anthropic.com/en/docs/claude-code](https://docs.anthropic.com/en/docs/claude-code) |
| **Primary models** | Claude Opus 4.7, Claude Sonnet 4.6 (configurable) |
| **Skill system** | `~/.claude/skills/<name>/SKILL.md` — slash commands invoked by name |
| **Hook system** | PreToolUse / PostToolUse / SessionStart / SessionEnd events |

---

## Role in This Knowledge Base

Claude Code is the **agent host** for the entire "Harness Engineering" vault:
- Reads `CLAUDE.md` schema at session start, loads `wiki/index.md`
- Executes all Ingest / Query / Lint workflows defined in CLAUDE.md §3
- Runs MinerU skill for PDF→Markdown conversion
- Creates and maintains all pages under `wiki/`
- Invokes specialized sub-skills (mineru, paper-reader, research-wiki, etc.) via the Skill tool

---

## MCP Ecosystem

Claude Code's Model Context Protocol allows external tools to appear as native capabilities:
- **Codex MCP** (`mcp__codex__*`) — routes reviewer calls to GPT-5.5 xhigh; core to ARIS's cross-model review
- **Zotero MCP** — paper import, collection management, citation export
- **Google Drive MCP** — cloud document access
- **MinerU MCP** — PDF/DOCX/PPTX parsing to Markdown
- **llm-chat MCP** — any OpenAI-compatible API as reviewer (Kimi, GLM, MiniMax, DeepSeek)

---

## Skill Architecture

Skills are plain Markdown files (`SKILL.md`) invoked via the `Skill` tool. ARIS ships 65+ skills organized into:
- Research pipeline skills (idea-discovery, novelty-check, auto-review-loop, paper-writing, rebuttal)
- Audit/assurance skills (experiment-audit, result-to-claim, paper-claim-audit, citation-audit)
- Utility skills (mineru, research-wiki, overleaf-sync, meta-optimize)

The `superpowers` plugin (active in this vault) adds its own layer of skills for task management, planning, and workflow orchestration.

---

## Why Claude Code as ARIS Host

- **Speed × autonomy**: fluid tool use (file reads, bash, web fetch) without per-action approval delays
- **Skill system**: slash commands map 1:1 to SKILL.md files — no framework overhead
- **Hook system**: session lifecycle events (start/end/stop) enable automatic logging and meta-optimization
- **MCP flexibility**: swap executor or reviewer backend without changing skill prompts

---

## Related Pages

- [[wiki/entities/frameworks/ARIS]] — harness built on Claude Code
