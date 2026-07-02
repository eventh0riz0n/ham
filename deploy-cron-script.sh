#!/usr/bin/env bash
# Weekly HAM v2 hygiene: dedup near-duplicate facts, decay stale importance,
# prune long-superseded rows. Deterministic — no LLM. Replaces the v1 pair
# (HAM Daily Indexer + HAM Weekly Consolidation), both paused 2026-07-02.
set -euo pipefail
HERMES="/home/ben/.local/bin/hermes"
out="$($HERMES ham consolidate 2>&1 | tail -5)"
stats="$($HERMES ham status 2>&1 | tail -20)"
echo "HAM v2 weekly consolidate: ${out}"
echo "${stats}"
