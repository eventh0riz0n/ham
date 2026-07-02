# Changelog

## 2.0.0 - 2026-07-02

Ground-up rewrite as a **native Hermes MemoryProvider plugin**. v1 was a
standalone retrieval engine that was never wired into the agent runtime —
nothing read from it at conversation time and no sessions were written to it.
v2 plugs into the host's memory lifecycle instead. The v1 package remains
available at tag `v1.0.0`.

### Added
- MemoryProvider integration: per-turn hybrid recall injected as
  `<memory-context>`, turn buffering, extraction at session end /
  pre-compress / `/reset`, mirroring of built-in MEMORY.md/USER.md writes.
- Write path with reconciliation: one LLM call per session boundary emits
  explicit `add` / `update` / `supersede` / `noop` operations against the
  nearest existing facts (Mem0-style), so contradictions and duplicates are
  resolved at write time.
- Temporal fact model: `status`, `superseded_by`, `invalidated_at` — facts
  are superseded, never silently deleted; history stays queryable.
- Episode summaries (1 per session) carrying a `session_id` pointer for deep
  transcript recall via the host's `session_search`.
- `ham_memory` agent tool (recall / remember / forget / status) and
  `hermes ham` CLI (status / recall / remember / forget / list /
  consolidate / reembed).
- Deterministic weekly consolidation: near-duplicate merge, importance decay,
  pruning of long-superseded rows. No LLM.
- Polish-friendly FTS (unicode61 `remove_diacritics 2`) and a multilingual
  local embedding model (PL+EN).

### Changed
- Facts, not chunks: no file/skill indexing (v1's corpus was 99% SKILL.md
  dumps that crowded out fact recall).
- Single local embedding space (fastembed
  `paraphrase-multilingual-MiniLM-L12-v2`, 384d). No remote embedding APIs,
  no cross-provider vector spaces, no hash-fallback pseudo-vectors — if the
  embedder is unavailable, search degrades to BM25.
- Embeddings stored as BLOBs with brute-force numpy cosine; sqlite-vec
  dropped (its never-reclaimed vector chunks bloated a v1 production DB to
  333 MB for 4.8 MB of text).

### Removed
- Indexer (`ham index`), session hook module, embedding provider fallback
  chain, projection of foreign-dimension vectors, standalone `pyproject`
  packaging.

## 1.0.0 - 2026-04-27

### Added
- Provider-aware vector retrieval guard with per-chunk `embedding_provider` metadata.
- Schema migration v2 for existing databases.
- CLI commands: `list`, `show`, `delete`, `backup`, `doctor`, `migrate`.
- `ham index --dry-run` and repeatable `--exclude` patterns.
- Conservative indexer denylist for secrets, private keys, caches, and DB files.
- GitHub Actions CI for Python 3.10, 3.11, and 3.12.
- LICENSE, docs, and examples for 1.0 usage.

### Changed
- Minimal installs work without external API keys or optional local embedding packages.
- FTS queries are tokenized/quoted and robust against reserved-word syntax errors.
- CLI errors are friendlier for invalid `--since` values.

### Fixed
- Optional FastEmbed is now lazy-loaded.
- `HAM_DB_PATH` is respected by package and CLI usage.
- CLI `remember` and `recall` output now reflects actual return shapes.
