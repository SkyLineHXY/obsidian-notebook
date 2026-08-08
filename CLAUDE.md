# Wiki Schema — Compact Claude Guide

**Start every session by reading this file, then `wiki/index.md`.** Keep `index.md` and `log.md` compact; open detailed pages only when needed.

## Structure & Permissions

- `raw/`: user-owned, read-only original materials.
  - Assets: `raw/assets/`
  - Parsed sources: `raw/sources/`
- `wiki/`: Claude-owned, read/write knowledge base.
  - `index.md`, `log.md`, `overview.md`
  - `sources/`, `entities/`, `concepts/`, `comparisons/`, `analyses/`
- `CLAUDE.md`: shared schema/config.

Directory conventions:
- Entities: `wiki/entities/<models|frameworks|tools|hardware|systems|people>/`
- Concepts: `wiki/concepts/<rl|imitation-learning|generative-models|vla|benchmarks|infrastructure|slam>/`
- Sources: `wiki/sources/<theme>/YYYY-MM-DD 标题.md`
- Comparisons and analyses stay flat.

## Page Requirements

Every wiki page needs frontmatter:

```yaml
---
type: entity | concept | source | comparison | analysis
tags: [Tag1, Tag2]
sources: [文件路径或 URL]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Rules:
- Use Obsidian links: `[[wiki/.../Page]]`.
- Source pages must link back to raw material with clean wiki links, not code formatting.
- Paper figures from MinerU must reference raw images directly:
  `![[raw/sources/papers/<topic>/<stem>/images/<hash>.jpg]]`
- Math must use LaTeX: inline `$...$`, block `$$...$$`. Do not put formulas in code blocks.
- RL / IL / control / generative AI / optimization concept or analysis pages need `## 严谨数学推导`.
- Create source pages for every ingested source. Create entity/concept/comparison pages only when ≥2 independent sources justify them.

## Workflows

### Ingest
1. PDF/DOCX/PPTX must be parsed by MinerU first.
2. Read parsed Markdown; inspect 1-3 key figures when useful.
3. Discuss key takeaways with the user when ambiguity matters.
4. Write source page with frontmatter, raw back-link, and key figure links.
5. Upgrade recurring entities/concepts if ≥2 sources.
6. Update relevant pages + compact `wiki/index.md`.
7. Append a short entry to `wiki/log.md`.
8. Git sync when permissions allow.

### Query
1. Read `wiki/index.md`.
2. Open only relevant pages.
3. Answer with cited wiki page references.
4. If a reusable insight is produced, create/update comparison or analysis page, then update index/log.

### Lint
Check frontmatter, broken wiki links, orphan pages, stale index counts, formula style, raw back-links, image links, and recent unparsed PDFs.

## MinerU Notes

- Use MinerU for PDF/DOCX/PPTX before wiki writing.
- Output mirror: `raw/assets/papers/<topic>/<file>.pdf` → `raw/sources/papers/<topic>/<stem>/<stem>.md` + `images/`.
- `MINERU_TOKEN` is in `~/.claude/settings.json`; do not pass `--token`.
- Use Windows `python`, not `python3`.

## Git Sync

Repo: `https://github.com/SkyLineHXY/obsidian-notebook.git`, branch `main`.

```bash
git add wiki/ CLAUDE.md raw/sources/
git commit -m "docs(wiki): <一句话描述>"
git -c http.proxy="" -c https.proxy="" push origin main
```

Push must clear local proxy as above.

## Token Budget Policy

- Keep `wiki/index.md` as a navigation map, not a full catalog.
- Keep `wiki/log.md` as compressed milestones; each new entry should be 1-3 bullets.
- Put detailed reasoning inside dedicated source/concept/analysis pages, not in index/log.

> Schema compacted: 2026-06-08.
