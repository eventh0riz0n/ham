---
name: advanced-memory
title: Advanced Memory (HAM) 🍖
version: 1.0
category: memory
description: Multi-store local memory with hybrid retrieval (BM25 + vector + recency + importance). Better than OpenClaw.
---

# Hermes Advanced Memory (HAM) 🍖

## Why Better Than OpenClaw

| Feature | OpenClaw | HAM |
|---|---|---|
| Search | FTS5 + vector (flat) | **Hybrid**: BM25 + vector + recency + importance |
| Stores | Single flat index | **Multi-store**: episodic, semantic, procedural, archive |
| Auto-extract | No | **Yes**: auto-extract facts from sessions |
| Consolidation | No | **Yes**: auto-merge old episodes, deduplicate semantic |
| Links | No | **Yes**: semantic links between chunks |
| Self-evaluation | No | **Yes**: access-count feedback loop boosts popular memories |
| Importance | Static | **Dynamic**: adjustable per-chunk |

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

## Stores

### Episodic (wydarzenia)
- Raw conversations, sessions
- Auto-consolidated to archive after 7 days if rarely accessed

### Semantic (wiedza)
- Facts, preferences, decisions
- Auto-deduplicated
- User profile, configurations

### Procedural (jak robić)
- Skills, workflows, scripts
- How-to knowledge

### Archive (skompresowane)
- Old episodic summaries
- Rarely accessed but preserved

## Commands

```bash
# Index everything
ham index --full

# Remember something
ham remember "User prefers dark mode" --store semantic --importance 0.8

# Recall
ham recall "what did user say about crypto"

# Stats
ham stats

# Consolidate old memories
ham consolidate

# Get context for a prompt (Python)
python -c "from ham.session_hook import get_context_for_prompt; print(get_context_for_prompt('planning a trip'))"
```

## Files
- `ham/memory_engine.py` — core engine (SQLite + sqlite-vec + FTS5)
- `ham/indexer.py` — auto-indexes memory files, workspace docs, skills
- `ham/session_hook.py` — saves sessions, extracts facts, retrieves context
- `ham/cli.py` — command-line interface

## Database
`~/.hermes/memory/ham.db` — single SQLite file with:
- `chunks` — all memory chunks
- `chunks_fts` — FTS5 virtual table
- `chunks_vec` — sqlite-vec virtual table
- `embedding_cache` — deduped embeddings
- `semantic_links` — graph relationships
- `consolidation_log` — auto-maintenance log
- `schema_version` — migration tracking

## Embedding Provider (Multi-Tier Fallback)

HAM uses a **cascading provider chain** with automatic failover, retry, and local backup. No single point of failure.

```
Request
  → Gemini embedding-001 (3072 dims)
      ↓ 429 / timeout / fail
      Retry 3× with exponential backoff (1s → 3s → 7s)
  → OpenRouter text-embedding-3-large (3072 dims, batch API)
  → FastEmbed local ONNX (BAAI/bge-small-en-v1.5, 384 dims)
      ↓ projected to 3072 via seeded random Gaussian
  → Hash fallback (degraded, preserves system continuity)
```

### Features

| Feature | Implementation |
|---|---|
| **Retry** | 3 attempts per provider with exponential backoff on 429/5xx |
| **Batch** | `add_chunk()` embeds all chunks in one call (OpenRouter/FastEmbed support batch) |
| **Cache** | SHA256 dedup cache — never pay twice for the same text |
| **Local** | FastEmbed (~30MB ONNX) runs fully offline on CPU |
| **Projection** | Any dimension → 3072 via seeded random matrix; preserves cosine angles |
| **Zero-downtime** | If all providers fail, hash fallback returns deterministic vectors so search still works |

### Requirements

```bash
# FastEmbed (local fallback) — installs in ~5s, no torch
pip install fastembed
```

### Environment

```bash
# Primary (recommended — free tier, 1500 RPM)
GEMINI_API_KEY=***

# Secondary (optional — paid, batch-capable)
OPENROUTER_API_KEY=***

# Tertiary (local — always works, zero API calls)
# FastEmbed auto-downloads model on first use
```

### ⚠️ Mixed Embedding Space Trap

**Critical bug you will hit:** projecting a 384-dim FastEmbed vector to 3072 dims does **NOT** make it compatible with Gemini 3072-dim vectors in the same DB. Cross-provider cosine similarity collapses to ~0 (random).

```
Gemini vs Gemini:           0.7285  ✓
FastEmbed vs FastEmbed:     0.7409  ✓
Gemini vs FastEmbed (proj): -0.0140  ✗ CATASTROPHE
```

**Fix:** provider-aware vector guard — disable vector search when query provider ≠ DB provider, fall back to BM25 + recency only:

```python
can_vector = (query_provider == db_provider)
if can_vector:
    score = 0.4*vector + 0.3*BM25 + 0.2*recency + 0.1*importance
else:
    score = 0.55*BM25 + 0.35*recency + 0.1*importance  # vector OFF
```

This preserves retrieval quality instead of returning noise.

### Provider selection per query

```python
from ham.memory_engine import MemoryEngine
eng = MemoryEngine()

# Returns (embeddings, provider_name)
emb, provider = eng.embedding_mgr.get_embeddings([text])
print(provider)  # "gemini", "openrouter", "fastembed", "hash", or "cache"
```

## Auto-Maintenance
Two cron jobs recommended:

**Daily indexer** (4:00 AM) — indexes new/updated memory files, skills, workspace docs:
```bash
ham index --full
```

**Weekly consolidation** (Sunday 3:00 AM) — compresses old episodic memory and deduplicates semantic facts:
```bash
ham consolidate
```

## Integration
To use in a session, retrieve context before generating response:
```python
from ham.session_hook import get_context_for_prompt
context = get_context_for_prompt(user_message, top_k=5)
# prepend context to system prompt
```
