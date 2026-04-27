# HAM 🍖 — Hermes Advanced Memory

Local-first, multi-store memory system for AI agents. Hybrid retrieval (BM25 + vector + recency + importance) with automatic provider fallback. Zero external dependencies required.

> **Better than OpenClaw.** Smarter search. Multiple memory stores. Self-maintaining.

---

## Why

Most agent memory is a flat vector dump. HAM is a **structured, self-improving memory layer** that actually retrieves the right context.

| Feature | OpenClaw | HAM |
|---|---|---|
| Search | FTS5 + vector (flat) | **Hybrid**: BM25 + vector + recency + importance |
| Stores | Single flat index | **Multi-store**: episodic, semantic, procedural, archive |
| Auto-extract | No | **Yes**: auto-extract facts from sessions |
| Consolidation | No | **Yes**: auto-merge old episodes, deduplicate semantic |
| Links | No | **Yes**: semantic links between chunks |
| Self-evaluation | No | **Yes**: access-count feedback loop boosts popular memories |
| Importance | Static | **Dynamic**: adjustable per-chunk |

---

## Architecture

```
User conversation
       |
       v
[Session Hook] ---> saves to episodic memory
       |
       +---> [Fact Extractor] ---> saves to semantic memory
       |
       v
[Hybrid Retriever] <--- query from next conversation
       |
       +---> BM25 (keyword)
       +---> Vector (semantic)
       +---> Recency (time decay)
       +---> Importance (manual/auto)
       +---> Access frequency (feedback loop)
       |
       v
[Context Assembly] ---> prepended to prompt
```

---

## Quick Start

```bash
# Install dependencies
uv pip install sqlite-vec fastembed

# 1. Index your workspace
python scripts/indexer.py --full

# 2. Remember something
python scripts/memory_engine.py -r "User prefers dark mode" -s semantic --importance 0.8

# 3. Recall
python scripts/memory_engine.py -q "what did user say about crypto"

# 4. Stats
python scripts/memory_engine.py --stats

# 5. Consolidate old memories (weekly cron recommended)
python scripts/memory_engine.py --consolidate
```

---

## Memory Stores

| Store | Content | Lifecycle |
|---|---|---|
| **Episodic** | Raw conversations, sessions | Auto-consolidated to archive after 7 days if rarely accessed |
| **Semantic** | Facts, preferences, decisions | Auto-deduplicated |
| **Procedural** | Skills, workflows, scripts | Persistent |
| **Archive** | Old episodic summaries | Rarely accessed but preserved |

---

## Multi-Tier Embedding Fallback

No single point of failure. If a provider hits limits, HAM automatically falls back.

```
Request
  → Gemini embedding-001 (3072 dims, free tier 1500 RPM)
      ↓ 429 / timeout / fail
      Retry 3× with exponential backoff (1s → 3s → 7s)
  → OpenRouter text-embedding-3-large (3072 dims, batch API)
  → FastEmbed local ONNX (BAAI/bge-small-en-v1.5, 384 dims)
      ↓ projected to 3072 via seeded random Gaussian
  → Hash fallback (degraded, preserves system continuity)
```

**Features:**
- **Retry**: 3 attempts per provider with exponential backoff on 429/5xx
- **Batch**: `add_chunk()` embeds all chunks in one call
- **Cache**: SHA256 dedup cache — never pay twice for the same text
- **Local**: FastEmbed (~30MB ONNX) runs fully offline on CPU
- **Zero-downtime**: If all providers fail, hash fallback returns deterministic vectors

### Environment Variables

```bash
# Primary (recommended — free tier, 1500 RPM)
GEMINI_API_KEY=***

# Secondary (optional — paid, batch-capable)
OPENROUTER_API_KEY=***

# Tertiary (local — always works, zero API calls)
# FastEmbed auto-downloads model on first use
```

### ⚠️ Mixed Embedding Space Guard

Projecting a 384-dim FastEmbed vector to 3072 dims does **NOT** make it compatible with Gemini 3072-dim vectors. Cross-provider cosine similarity collapses to ~0.

HAM detects this and **automatically disables vector search** when providers mismatch, falling back to BM25 + recency instead of returning noise.

---

## Files

| File | Purpose |
|---|---|
| `scripts/memory_engine.py` | Core engine (SQLite + sqlite-vec + FTS5) |
| `scripts/indexer.py` | Auto-indexes memory files, workspace docs, skills |
| `scripts/session_hook.py` | Saves sessions, extracts facts, retrieves context |

---

## Database Schema

Single SQLite file (`~/.hermes/memory/ham.db` or set `HAM_DB_PATH`):

- `chunks` — all memory chunks
- `chunks_fts` — FTS5 virtual table
- `chunks_vec` — sqlite-vec virtual table (3072 dims)
- `embedding_cache` — deduped embeddings with provider tracking
- `semantic_links` — graph relationships
- `consolidation_log` — auto-maintenance log

---

## Integration

```python
from memory_engine import MemoryEngine
from session_hook import get_context_for_prompt

# Retrieve relevant context for a prompt
context = get_context_for_prompt("planning a trip", top_k=5)
# prepend context to system prompt
```

---

## Cron Jobs (Recommended)

**Daily indexer** (4:00 AM):
```bash
python scripts/indexer.py --full
```

**Weekly consolidation** (Sunday 3:00 AM):
```bash
python scripts/memory_engine.py --consolidate
```

---

## Status

Work in progress. Core engine is solid and battle-tested. Missing before 1.0:
- [ ] Unit tests
- [ ] `pyproject.toml`
- [ ] CLI (`ham recall/remember`)
- [ ] Native Hermes session hook integration
- [ ] Schema versioning

PRs welcome.
