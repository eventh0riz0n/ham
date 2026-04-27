# Privacy and Safety

HAM is local-first and stores memory in a local SQLite database. Treat that database as private data.

## What not to commit

Never commit:

- `ham.db`
- backups of `ham.db`
- `.env`
- API keys/tokens
- SSH keys or private certificates

## Indexer denylist

By default the indexer skips common sensitive files and caches:

- `.env`, `.env.*`
- `id_rsa`, `id_ed25519`
- `*.pem`, `*.key`
- `credentials.json`, `token.json`
- `.git`, `__pycache__`, `.pytest_cache`
- `*.db`, `*.sqlite`, logs, temp/backup files

Preview before indexing:

```bash
ham index --dry-run
```

Add custom excludes:

```bash
ham index --exclude 'client-secrets' --exclude 'private-notes'
```

## External services

HAM can use Gemini or OpenRouter if configured, but does not require them. Without keys it falls back to local/ deterministic providers. If privacy is more important than embedding quality, do not set external API keys and install/use local embedding support instead.
