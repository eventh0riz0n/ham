# HAM v2 — Hermes Advanced Memory

Native Hermes **MemoryProvider plugin**: local hybrid memory with a temporal
fact store, per-turn recall injection, and automatic LLM fact extraction at
session boundaries. 100% local storage, local embeddings.

Built 2026-07-02 as a ground-up replacement for HAM v1 (a standalone
retrieval engine that was never wired into the agent runtime).

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
| `store.py` | SQLite fact store: FTS5 (unicode61, diacritics-insensitive) + embedding BLOBs with brute-force numpy cosine; hybrid scoring; consolidate/reembed maintenance |
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
