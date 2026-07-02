# Privacy and Safety

HAM v2 is local-only. Stronger claims than v1:

- **No network calls, ever.** v1 could send text to Gemini/OpenRouter for
  embeddings; v2 embeds exclusively with a local fastembed (ONNX) model.
  Fact extraction runs through the *host's* LLM client — the same model and
  auth the user already configured for Hermes — HAM itself holds no keys
  and opens no connections.
- **No filesystem indexing.** v1 crawled memory files, workspace docs, and
  skills (with a denylist for secrets). v2 has no indexer at all — the only
  write paths are the conversation itself (extraction), explicit
  `remember` calls, and mirrored built-in memory writes. The denylist
  problem disappears with the crawler.
- **Extraction guardrails.** The extraction prompt forbids storing secrets,
  tokens, and passwords; operations are size- and count-capped; unknown
  fact ids are ignored; nothing is ever hard-deleted by the LLM
  (supersede keeps history, weekly pruning is deterministic host-side code).
- **Cron/subagent isolation.** Sessions with a non-primary agent context are
  never extracted, so automation transcripts cannot pollute user facts.

## Treat the database as private data

Everything the user tells the agent may end up in
`$HERMES_HOME/memory/ham_v2.db`. Never commit:

- `ham_v2.db` (or any `*.db` / WAL / SHM files)
- backups of the database
- `.env`, API keys, tokens, private keys

The repository `.gitignore` in this plugin covers Python artifacts only —
the database lives outside the repo tree (`$HERMES_HOME/memory/`), which is
the primary safeguard.

## Reviewing what is stored

```bash
hermes ham list --all          # every fact, including superseded history
hermes ham forget <fact_id>    # mark a fact as no longer true
hermes ham consolidate         # prune long-superseded rows now
```

`hermes backup` includes the database; treat backup archives with the same
care as the live file.
