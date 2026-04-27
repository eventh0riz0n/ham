#!/usr/bin/env python3
"""CLI for HAM 🍖"""

import argparse
import json
import sys
from pathlib import Path

from ham.memory_engine import MemoryEngine, DB_PATH
from ham.indexer import run_full_index


def cmd_recall(args):
    engine = MemoryEngine(DB_PATH)
    results = engine.recall(args.query, store=args.store, top_k=args.top_k)
    engine.close()
    if not results:
        print("No memories found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r.get('score', 0):.3f}) ---")
        print(f"store: {r.get('store')} | source: {r.get('source')} | type: {r.get('source_type')}")
        print(f"created: {r.get('created_at')} | accessed: {r.get('accessed_at')} | importance: {r.get('importance')}")
        print(f"text:\n{r.get('text', '')[:500]}")


def cmd_remember(args):
    engine = MemoryEngine(DB_PATH)
    chunk_id = engine.remember(
        text=args.text,
        store=args.store,
        source=args.source or "cli",
        source_type=args.source_type or "manual",
        importance=args.importance,
    )
    engine.close()
    print(f"Remembered as chunk {chunk_id}")


def cmd_stats(args):
    engine = MemoryEngine(DB_PATH)
    stats = engine.get_stats()
    engine.close()
    print(json.dumps(stats, indent=2, default=str))


def cmd_consolidate(args):
    engine = MemoryEngine(DB_PATH)
    print("Consolidating episodic memory...")
    engine.consolidate_episodic(days=args.days)
    print("Deduplicating semantic memory...")
    engine.deduplicate_semantic(threshold=args.threshold)
    engine.close()
    print("Done.")


def cmd_index(args):
    run_full_index()


def main():
    parser = argparse.ArgumentParser(prog="ham", description="HAM 🍖 — Hermes Advanced Memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # recall
    p_recall = sub.add_parser("recall", help="Search memory")
    p_recall.add_argument("query", help="Search query")
    p_recall.add_argument("--store", choices=["episodic", "semantic", "procedural", "archive"], default=None)
    p_recall.add_argument("--top-k", type=int, default=5)
    p_recall.set_defaults(func=cmd_recall)

    # remember
    p_rem = sub.add_parser("remember", help="Save a memory")
    p_rem.add_argument("text", help="Text to remember")
    p_rem.add_argument("--store", choices=["episodic", "semantic", "procedural"], default="semantic")
    p_rem.add_argument("--source", default="cli")
    p_rem.add_argument("--source-type", default="manual")
    p_rem.add_argument("--importance", type=float, default=0.5)
    p_rem.set_defaults(func=cmd_remember)

    # stats
    p_stats = sub.add_parser("stats", help="Show database stats")
    p_stats.set_defaults(func=cmd_stats)

    # consolidate
    p_con = sub.add_parser("consolidate", help="Consolidate old memories")
    p_con.add_argument("--days", type=int, default=7)
    p_con.add_argument("--threshold", type=float, default=0.92)
    p_con.set_defaults(func=cmd_consolidate)

    # index
    p_idx = sub.add_parser("index", help="Index files into memory")
    p_idx.add_argument("--full", action="store_true", help="Full re-index")
    p_idx.set_defaults(func=cmd_index)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
