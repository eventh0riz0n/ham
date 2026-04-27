# Installation

## Minimal local install

HAM works without external API services. With no API keys, it falls back to deterministic hash embeddings so the system remains usable.

```bash
pip install git+https://github.com/eventh0riz0n/ham.git
ham --help
```

For local development from a clone:

```bash
git clone https://github.com/eventh0riz0n/ham.git
cd ham
pip install -e ".[dev]"
```

## Optional extras

```bash
pip install -e ".[local]"       # FastEmbed local CPU embeddings
pip install -e ".[openrouter]"  # OpenRouter/OpenAI-compatible fallback
pip install -e ".[all]"         # everything + dev tools
```

## Database path

Default DB:

```text
~/.hermes/memory/ham.db
```

Override for tests/smoke checks:

```bash
HAM_DB_PATH=/tmp/ham.db ham remember "hello" --store semantic
```

## Verify install

```bash
HAM_DB_PATH=/tmp/ham-smoke.db ham remember "install smoke" --store semantic
HAM_DB_PATH=/tmp/ham-smoke.db ham recall "install smoke"
HAM_DB_PATH=/tmp/ham-smoke.db ham doctor
```
