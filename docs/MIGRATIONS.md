# Migrations

HAM tracks schema migrations in the `schema_version` table.

Current schema target: `2`.

## Commands

```bash
ham migrate --dry-run
ham migrate
```

Opening `MemoryEngine()` also applies pending migrations.

## Backup before major upgrades

```bash
ham backup --out ~/ham-before-upgrade.db
# If the file exists, choose a new path or pass --force intentionally.
ham migrate
ham doctor
```

## v2

Adds `chunks.embedding_provider` so vector retrieval can avoid comparing incompatible embedding spaces.

Existing rows are marked as `unknown`. They remain searchable via BM25/recency/importance, but are excluded from provider-matched vector search until re-indexed/re-remembered.
