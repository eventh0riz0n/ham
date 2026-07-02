"""CLI commands for HAM v2: hermes ham status|recall|remember|forget|list|consolidate|reembed

Registered automatically while ``memory.provider: ham`` is active.
The weekly maintenance cron calls ``hermes ham consolidate``.
"""

from __future__ import annotations

import json
from pathlib import Path


def _open_store():
    # Import store.py directly — the CLI registration path loads this module
    # under a synthetic package shell whose __init__.py never executed, so
    # `from . import <name-defined-in-__init__>` is not available here.
    from .store import HamStore, DEFAULT_EMBED_MODEL
    from hermes_constants import get_hermes_home

    hermes_home = str(get_hermes_home())
    cfg = {}
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = cfg_get(load_config() or {}, "plugins", "ham", default={}) or {}
    except Exception:
        pass
    raw = str(cfg.get("db_path") or "")
    if raw:
        raw = raw.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        db_path = Path(raw).expanduser()
    else:
        db_path = Path(hermes_home) / "memory" / "ham_v2.db"
    embed_model = str(cfg.get("embed_model") or DEFAULT_EMBED_MODEL)
    return HamStore(db_path, embed_model=embed_model)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def ham_command(args) -> None:
    sub = getattr(args, "ham_command", None)
    store = _open_store()
    try:
        if sub in (None, "status"):
            _print(store.stats())
        elif sub == "recall":
            results = store.search(
                " ".join(args.query),
                top_k=args.top_k,
                include_superseded=args.all,
                include_episodes=True,
                touch=False,
            )
            for r in results:
                flag = "" if r["status"] == "active" else " [SUPERSEDED]"
                print(f"#{r['id']} [{r['score']}] ({r['kind']}){flag} {r['text']}")
            if not results:
                print("(no results)")
        elif sub == "remember":
            fid = store.add_fact(
                " ".join(args.text), kind=args.kind,
                importance=args.importance, source="cli",
            )
            print(f"stored fact #{fid}")
        elif sub == "forget":
            ok = store.invalidate(args.fact_id, reason=args.reason or "cli")
            print("forgotten (kept as history)" if ok else "fact not found")
        elif sub == "list":
            for r in store.list_facts(status="all" if args.all else "active",
                                      kind=args.kind, limit=args.limit):
                flag = "" if r["status"] == "active" else " [SUPERSEDED]"
                print(f"#{r['id']} ({r['kind']}, imp {r['importance']:.2f}){flag} {r['text'][:160]}")
        elif sub == "consolidate":
            _print(store.consolidate())
        elif sub == "reembed":
            print(f"re-embedded {store.reembed_missing()} facts")
        else:
            print(f"Unknown ham command: {sub}")
    finally:
        store.close()


def register_cli(subparser) -> None:
    subs = subparser.add_subparsers(dest="ham_command")

    subs.add_parser("status", help="Store statistics")

    p = subs.add_parser("recall", help="Hybrid search over facts")
    p.add_argument("query", nargs="+")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--all", action="store_true", help="Include superseded facts")

    p = subs.add_parser("remember", help="Store a durable fact")
    p.add_argument("text", nargs="+")
    p.add_argument("--kind", default="note",
                   choices=["user_pref", "project", "infra", "decision", "note"])
    p.add_argument("--importance", type=float, default=0.6)

    p = subs.add_parser("forget", help="Mark a fact as no longer true")
    p.add_argument("fact_id", type=int)
    p.add_argument("--reason", default="")

    p = subs.add_parser("list", help="List facts")
    p.add_argument("--kind", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--all", action="store_true", help="Include superseded facts")

    subs.add_parser("consolidate", help="Weekly hygiene: dedup, decay, prune (no LLM)")
    subs.add_parser("reembed", help="Embed facts missing current-model vectors")
