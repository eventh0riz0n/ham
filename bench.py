#!/usr/bin/env python3
"""Recall benchmark for HAM prefetch quality.

Runs labeled real-conversation turns against a frozen DB snapshot and
reports, per policy:

  recall@K     — of the facts a turn genuinely needed, how many were injected
  precision@K  — of the facts injected, how many were genuinely needed
  clean-noise  — facts injected on turns where NOTHING should be injected

The dataset lives OUTSIDE the repo (it contains private conversation turns):
  ~/.hermes/memory/ham_bench.json   {"db": path, "cases": [
      {"prev_user":…, "prev_asst":…, "user":…, "relevant":[fact ids]}]}

Policies compared:
  baseline — v2.0 behavior: query = user message alone, gate = total
             score >= 0.35 (recency/importance leak through the gate)
  current  — whatever the working tree implements: windowed query for
             short messages + match-score gate

Usage: python bench.py [--min-match X] [--top-k N]
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from store import HamStore  # noqa: E402


def _build_prefetch_query():
    """Import the provider's query builder, stubbing the host if absent."""
    try:
        import agent.memory_provider  # noqa: F401
    except ImportError:
        agent_mod = types.ModuleType("agent")
        mp_mod = types.ModuleType("agent.memory_provider")
        mp_mod.MemoryProvider = type("MemoryProvider", (), {})
        agent_mod.memory_provider = mp_mod
        sys.modules.setdefault("agent", agent_mod)
        sys.modules.setdefault("agent.memory_provider", mp_mod)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import ham
    return ham.build_prefetch_query

DATASET = Path("~/.hermes/memory/ham_bench.json").expanduser()
TOP_K = 4


def load_dataset():
    data = json.loads(DATASET.read_text())
    return Path(data["db"]).expanduser(), data["cases"]


def run_policy(store, cases, *, top_k, select):
    """select(store, case, top_k) -> list of injected fact ids."""
    recalls, precisions = [], []
    clean_noise = clean_turns = 0
    for case in cases:
        injected = select(store, case, top_k)
        relevant = set(case["relevant"])
        if relevant:
            hit = len(relevant & set(injected))
            recalls.append(hit / len(relevant))
            precisions.append(hit / len(injected) if injected else 1.0)
        else:
            clean_turns += 1
            clean_noise += len(injected)
    n = len(recalls)
    return {
        "recall": sum(recalls) / n if n else 0.0,
        "precision": sum(precisions) / n if n else 0.0,
        "clean_noise_per_turn": clean_noise / clean_turns if clean_turns else 0.0,
    }


def baseline_select(store, case, top_k):
    # v2.0: narrow query, order by weighted score, gate on total score.
    results = store.search(case["user"], top_k=top_k * 4, touch=False)
    results.sort(key=lambda r: r["score"], reverse=True)
    return [r["id"] for r in results[:top_k] if r["score"] >= 0.35]


def make_current_select(min_match):
    build_query = _build_prefetch_query()

    def _select(store, case, top_k):
        query = build_query(case["user"], case["prev_user"], case["prev_asst"])
        if not query:
            return []
        results = store.search(query, top_k=top_k, touch=False)
        return [r["id"] for r in results if r.get("match_score", 1.0) >= min_match]
    return _select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-match", type=float, default=None,
                    help="gate for 'current'; default: sweep and report best")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    db_path, cases = load_dataset()
    store = HamStore(db_path)
    store.embedder.warm()

    print(f"cases: {len(cases)}  (with-relevant: "
          f"{sum(1 for c in cases if c['relevant'])}, clean: "
          f"{sum(1 for c in cases if not c['relevant'])})  top_k={args.top_k}\n")

    b = run_policy(store, cases, top_k=args.top_k, select=baseline_select)
    print(f"{'baseline':<22} recall={b['recall']:.3f}  precision={b['precision']:.3f}"
          f"  clean-noise/turn={b['clean_noise_per_turn']:.2f}")

    gates = [args.min_match] if args.min_match is not None else \
        [0.40, 0.45, 0.50, 0.55, 0.60]
    for g in gates:
        c = run_policy(store, cases, top_k=args.top_k, select=make_current_select(g))
        print(f"{f'current(gate={g:.2f})':<22} recall={c['recall']:.3f}"
              f"  precision={c['precision']:.3f}"
              f"  clean-noise/turn={c['clean_noise_per_turn']:.2f}")


if __name__ == "__main__":
    main()
