# Usage

## Remember and recall

```bash
ham remember "User prefers concise Polish replies" --store semantic --importance 0.8
ham recall "reply language preference"
```

Stores:

- `episodic` — conversations/events
- `semantic` — durable facts/preferences/decisions
- `procedural` — workflows/how-to knowledge
- `archive` — consolidated old memory

## Manage chunks

```bash
ham list --store semantic --limit 20
ham show CHUNK_ID
ham delete CHUNK_ID
```

## Index local knowledge

```bash
ham index --dry-run
ham index --full
ham index --since 2026-04-27T10:00:00
ham index --exclude 'private-project'
```

The indexer has a conservative denylist for common secrets and cache files.

## Maintenance

```bash
ham doctor
ham backup --out ~/ham-backup.db
# Refuses to overwrite unless --force is passed.
ham migrate --dry-run
ham migrate
ham consolidate
ham stats
```
