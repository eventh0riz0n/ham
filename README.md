# HAM 🍖 — Hermes Advanced Memory

Local-first, multi-store memory system for AI agents. Hybrid retrieval (BM25 + vector + recency + importance) with automatic provider fallback. Zero external services required.

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
# Install from GitHub
pip install git+https://github.com/eventh0riz0n/ham.git

# Or install from a local clone
pip install -e .

# Or with local fallback support
pip install -e ".[local]"

# Or everything (local fallback + OpenRouter + dev tools)
pip install -e ".[all]"

# 1. Check health
ham doctor

# 2. Preview indexing without writing the DB
ham index --dry-run

# 3. Index your workspace
ham index --full

# 4. Remember something
ham remember "User prefers dark mode" --store semantic --importance 0.8

# 5. Recall
ham recall "what did user say about crypto"

# 6. Manage chunks
ham list --store semantic
ham show CHUNK_ID
ham delete CHUNK_ID

# 7. Backup / maintenance
ham backup --out ~/ham-backup.db
# Use --force only when you intentionally want to overwrite an existing backup.
ham consolidate
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
| `ham/memory_engine.py` | Core engine (SQLite + sqlite-vec + FTS5) |
| `ham/indexer.py` | Auto-indexes memory files, workspace docs, skills |
| `ham/session_hook.py` | Saves sessions, extracts facts, retrieves context |
| `ham/cli.py` | `ham` command-line interface: recall/remember/list/show/delete/doctor/backup/index |

---

## Database Schema

Single SQLite file (`~/.hermes/memory/ham.db` or set `HAM_DB_PATH`). Versioned migrations (see `schema_version` table).

| Table | Purpose |
|---|---|
| `chunks` | All memory chunks |
| `chunks_fts` | FTS5 virtual table |
| `chunks_vec` | sqlite-vec virtual table (3072 dims) |
| `embedding_cache` | Deduped embeddings with provider tracking |
| `semantic_links` | Graph relationships |
| `consolidation_log` | Auto-maintenance log |
| `schema_version` | Migration tracking |

---

## Integration

### Python API

```python
from ham.memory_engine import MemoryEngine
from ham.session_hook import get_context_for_prompt

engine = MemoryEngine()
engine.remember("User likes dark mode", store="semantic", importance=0.8)
results = engine.recall("dark mode", top_k=5)
```

### Hermes Skill

If you use Hermes Agent, the `advanced-memory` skill is auto-detected. Type `/advanced-memory` in chat to load this skill into context.

Session hook integration is manual for now — call `save_session()` at the end of conversations.

---

## Cron Jobs (Recommended)

**Daily indexer** (4:00 AM):
```bash
ham index --full
```

**Weekly consolidation** (Sunday 3:00 AM):
```bash
ham consolidate
```

---

## Development

```bash
# Run tests
pytest

# Install with dev deps
pip install -e ".[dev]"
```

---

## Status

| Feature | Status |
|---|---|
| Core engine (BM25 + vector hybrid) | ✅ Solid |
| Multi-tier embedding fallback | ✅ Done |
| Provider-aware vector guard | ✅ Done |
| Schema versioning | ✅ Done |
| CLI (`ham recall/remember/list/show/delete/stats/consolidate/index/doctor/backup/migrate`) | ✅ Done |
| Unit tests + CI | ✅ Done |
| `pyproject.toml` | ✅ Done |
| Native Hermes session hook | 🔄 Manual for now |

PRs welcome.
