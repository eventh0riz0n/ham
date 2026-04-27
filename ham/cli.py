#!/usr/bin/env python3
"""CLI for HAM 🍖"""

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime

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
        importance = r.get('breakdown', {}).get('importance', 'n/a')
        print(f"\n--- Result {i} (score: {r.get('score', 0):.3f}) ---")
        print(f"store: {r.get('store')} | source: {r.get('source')} | importance: {importance}")
        print(f"created: {r.get('created_at')} | access_count: {r.get('access_count')}")
        print(f"text:\n{r.get('text', '')[:500]}")


def cmd_remember(args):
    engine = MemoryEngine(DB_PATH)
    chunk_ids = engine.remember(
        text=args.text,
        store=args.store,
        source=args.source or "cli",
        source_type=args.source_type or "manual",
        importance=args.importance,
    )
    engine.close()
    print(f"Remembered {len(chunk_ids)} chunk(s):")
    for chunk_id in chunk_ids:
        print(f"  {chunk_id}")


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
    engine.deduplicate_semantic(similarity_threshold=args.threshold)
    engine.close()
    print("Done.")


def cmd_index(args):
    try:
        since = datetime.fromisoformat(args.since) if args.since else None
    except ValueError:
        raise SystemExit(f"Invalid --since value: {args.since!r}. Use ISO format, e.g. 2026-04-27T10:00:00")
    run_full_index(since=since, dry_run=args.dry_run, excludes=args.exclude)


def cmd_list(args):
    engine = MemoryEngine(DB_PATH)
    try:
        chunks = engine.list_chunks(store=args.store, limit=args.limit, offset=args.offset)
    finally:
        engine.close()
    for item in chunks:
        print(f"{item['id']} | {item['store']} | importance={item['importance']} | source={item['source']}")
        print(f"  {item['text'][:160]}")


def cmd_show(args):
    engine = MemoryEngine(DB_PATH)
    try:
        chunk = engine.get_chunk(args.chunk_id)
    finally:
        engine.close()
    if chunk is None:
        raise SystemExit(f"Chunk not found: {args.chunk_id}")
    print(json.dumps(chunk, indent=2, default=str))


def cmd_delete(args):
    engine = MemoryEngine(DB_PATH)
    try:
        deleted = engine.delete_chunk(args.chunk_id)
    finally:
        engine.close()
    if not deleted:
        raise SystemExit(f"Chunk not found: {args.chunk_id}")
    print(f"Deleted {args.chunk_id}")


def cmd_backup(args):
    engine = MemoryEngine(DB_PATH)
    try:
        out = engine.backup(Path(args.out), overwrite=args.force)
    except FileExistsError as e:
        raise SystemExit(str(e))
    finally:
        engine.close()
    print(f"Backup written to {out}")


def cmd_doctor(args):
    engine = MemoryEngine(DB_PATH)
    try:
        report = engine.doctor()
    finally:
        engine.close()
    print(json.dumps(report, indent=2, default=str))


def cmd_migrate(args):
    if args.dry_run:
        version = 0
        pending = [1, 2]
        if DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
            try:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
                ).fetchone()
                if row:
                    version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0
            finally:
                conn.close()
        pending = [v for v in pending if v > version]
        print(json.dumps({
            "db_path": str(DB_PATH),
            "current_schema_version": version,
            "target_schema_version": 2,
            "pending_migrations": pending,
            "dry_run": True,
        }, indent=2))
        return
    engine = MemoryEngine(DB_PATH)
    try:
        report = engine.doctor()
    finally:
        engine.close()
    print(json.dumps({"migrated": True, "schema_version": report["schema_version"]}, indent=2))


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
    p_idx.add_argument("--full", action="store_true", help="Full re-index (default behavior)")
    p_idx.add_argument("--since", help="Only index files modified after ISO datetime")
    p_idx.add_argument("--dry-run", action="store_true", help="Print files that would be indexed without writing DB")
    p_idx.add_argument("--exclude", action="append", default=[], help="Additional regex pattern to exclude (repeatable)")
    p_idx.set_defaults(func=cmd_index)

    # list
    p_list = sub.add_parser("list", help="List memory chunks")
    p_list.add_argument("--store", choices=["episodic", "semantic", "procedural", "archive"], default=None)
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--offset", type=int, default=0)
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = sub.add_parser("show", help="Show a memory chunk")
    p_show.add_argument("chunk_id")
    p_show.set_defaults(func=cmd_show)

    # delete
    p_delete = sub.add_parser("delete", help="Delete a memory chunk")
    p_delete.add_argument("chunk_id")
    p_delete.set_defaults(func=cmd_delete)

    # backup
    p_backup = sub.add_parser("backup", help="Backup the HAM database")
    p_backup.add_argument("--out", required=True, help="Output .db path")
    p_backup.add_argument("--force", action="store_true", help="Overwrite existing backup file")
    p_backup.set_defaults(func=cmd_backup)

    # doctor
    p_doctor = sub.add_parser("doctor", help="Run health checks")
    p_doctor.set_defaults(func=cmd_doctor)

    # migrate
    p_migrate = sub.add_parser("migrate", help="Run schema migrations")
    p_migrate.add_argument("--dry-run", action="store_true")
    p_migrate.set_defaults(func=cmd_migrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
