# Architecture

HAM is a local SQLite-backed memory layer for AI agents.

## Storage

- `chunks` — normalized memory records
- `chunks_fts` — FTS5 full-text index
- `chunks_vec` — sqlite-vec vector index
- `embedding_cache` — SHA256 text hash to embedding cache
- `semantic_links` — graph-like relations between chunks
- `schema_version` — migration history

## Retrieval

HAM combines:

- vector similarity when the query and chunk were embedded by the same provider
- BM25/FTS keyword search
- recency decay
- importance
- access-count feedback

## Provider-aware vector guard

Embedding providers can produce incompatible vector spaces. HAM stores `embedding_provider` on each chunk and only vector-compares chunks whose provider matches the query provider. Mismatched chunks can still rank via BM25, recency, and importance.

## Fallback providers

Provider chain:

1. Gemini, if configured
2. OpenRouter, if configured
3. FastEmbed, if installed
4. Hash fallback, always available

FastEmbed is lazy-loaded so minimal installs do not require the optional package.
