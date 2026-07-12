# HAM v2 — Hermes Advanced Memory

Native Hermes **MemoryProvider plugin**: local hybrid memory with a temporal
fact store, per-turn recall injection, and automatic LLM fact extraction at
session boundaries. 100% local storage, local embeddings.

Built 2026-07-02 as a ground-up replacement for HAM v1 (a standalone
retrieval engine that was never wired into the agent runtime). v2.1
(2026-07-12) reworked prefetch quality against a labeled benchmark of real
conversation turns: RRF ranking, a query-relatedness injection gate,
windowed queries for short messages, and injection dedup — see
`CHANGELOG.md` for the measured before/after.

## Retrieval

Per-turn prefetch builds a query from the user message (messages under
80 chars get the previous turn prepended — short chat messages carry no
retrieval signal on their own), then runs both lanes:

- **vector** — fastembed cosine over fact embeddings,
- **BM25** — SQLite FTS5, diacritics-insensitive,

fused with **Reciprocal Rank Fusion** (k=60). A fact is injected only if its
`match_score` — query-relatedness alone, no recency/importance — clears
`prefetch_min_match` (default 0.50, calibrated on the benchmark). Facts
injected within the last 3 turns are not re-injected. Recency and importance
still break ties and remain visible in results, but cannot push an unrelated
fact into the context.

`bench.py` measures recall/precision/noise of the whole prefetch path
against a labeled dataset of real turns. The dataset is **not** in the repo
(it contains private conversation content); it lives at
`~/.hermes/memory/ham_bench.json` with a frozen DB snapshot next to it.

## Install

1. Clone/copy this directory to `$HERMES_HOME/plugins/ham/` (usually
   `~/.hermes/plugins/ham/`). User plugins are discovered automatically.
2. Activate in `config.yaml`:

   ```yaml
   memory:
     provider: ham
   ```

3. (Optional) weekly hygiene cron — no-agent script calling
   `hermes ham consolidate` (see `../../scripts/ham-v2-consolidate.sh`):

   ```
   hermes cron create "30 3 * * 0" --name ham-v2-weekly-consolidate \
       --script ham-v2-consolidate.sh --no-agent --deliver local
   ```

First use downloads the fastembed model
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384d, PL+EN).

## Architecture

| File | Role |
|---|---|
| `__init__.py` | `HamMemoryProvider` — MemoryProvider lifecycle: prefetch per turn, turn buffering, extraction triggers, `ham_memory` tool |
| `store.py` | SQLite fact store: FTS5 (unicode61, diacritics-insensitive) + embedding BLOBs with brute-force numpy cosine; RRF hybrid ranking; consolidate/reembed maintenance |
| `bench.py` | Prefetch benchmark harness (recall@K / precision / clean-turn noise); dataset stays outside the repo |
| `extract.py` | Write path: one LLM call per session end — extract facts, reconcile against existing ones (add / update / supersede / noop), episode summary |
| `cli.py` | `hermes ham status\|recall\|remember\|forget\|list\|consolidate\|reembed` |
| `tests/` | pytest suite (fake embedder + fake LLM; no network) |

Design rationale (why no sqlite-vec, why one local embedding space, why
supersede-not-delete) is documented in the module docstrings of `store.py`
and `extract.py`.

## Data

DB: `$HERMES_HOME/memory/ham_v2.db` (override via `plugins.ham.db_path`).
Covered by `hermes backup` automatically (lives inside HERMES_HOME).

## Tests

```bash
cd ~/.hermes/hermes-agent && venv/bin/python -m pytest \
    ~/.hermes/plugins/ham/tests/ -q
```
