# Changelog

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
