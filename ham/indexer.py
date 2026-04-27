#!/usr/bin/env python3
"""
Indexer for HAM.
Indexes memory files, sessions, and workspace files into HAM.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from ham.memory_engine import MemoryEngine, DB_PATH

MEMORY_DIR = Path.home() / ".hermes" / "memory"
SESSION_SEARCH_DIR = Path.home() / ".hermes" / "sessions"
WORKSPACE_DIR = Path.home() / ".hermes"

# Files/patterns to skip
SKIP_PATTERNS = [
    r"\.db$", r"\.sqlite$", r"\.log$", r"\.jsonl$",
    r"brief_candidates\.json", r"feedback_log\.jsonl",
    r"ham\.db$", r"\.tmp$", r"\.bak$"
]


def should_skip(path: Path) -> bool:
    for pat in SKIP_PATTERNS:
        if re.search(pat, str(path)):
            return True
    return False


def index_memory_files(engine: MemoryEngine, since: datetime = None):
    """Index all markdown memory files."""
    indexed = 0
    for file_path in MEMORY_DIR.rglob("*.md"):
        if should_skip(file_path):
            continue
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if since and mtime < since:
                continue

            text = file_path.read_text(encoding="utf-8")
            if len(text.strip()) < 50:
                continue

            # Determine store type
            if "MEMORY" in file_path.name.upper():
                store = "semantic"
                importance = 0.7
            elif file_path.name.startswith("202"):  # daily note
                store = "episodic"
                importance = 0.5
            else:
                store = "semantic"
                importance = 0.5

            engine.add_chunk(
                text=text,
                store=store,
                source=str(file_path.relative_to(Path.home())),
                source_type="file",
                importance=importance,
                metadata={"file_mtime": mtime.isoformat(), "file_size": len(text)}
            )
            indexed += 1
        except Exception as e:
            print(f"[WARN] Failed to index {file_path}: {e}")

    return indexed


def index_workspace_docs(engine: MemoryEngine, since: datetime = None):
    """Index AGENTS.md, USER.md, SOUL.md, etc."""
    docs = ["AGENTS.md", "USER.md", "SOUL.md", "TOOLS.md", "HEARTBEAT.md"]
    indexed = 0
    for doc_name in docs:
        doc_path = WORKSPACE_DIR / doc_name
        if not doc_path.exists():
            continue
        try:
            mtime = datetime.fromtimestamp(doc_path.stat().st_mtime)
            if since and mtime < since:
                continue

            text = doc_path.read_text(encoding="utf-8")
            engine.add_chunk(
                text=text,
                store="semantic",
                source=str(doc_path.relative_to(Path.home())),
                source_type="workspace_doc",
                importance=0.9,  # Very important
                metadata={"doc_type": doc_name.replace(".md", "")}
            )
            indexed += 1
        except Exception as e:
            print(f"[WARN] Failed to index {doc_name}: {e}")
    return indexed


def index_skills(engine: MemoryEngine, since: datetime = None):
    """Index all SKILL.md files."""
    skills_dir = WORKSPACE_DIR / "skills"
    if not skills_dir.exists():
        return 0

    indexed = 0
    for skill_file in skills_dir.rglob("SKILL.md"):
        if should_skip(skill_file):
            continue
        try:
            mtime = datetime.fromtimestamp(skill_file.stat().st_mtime)
            if since and mtime < since:
                continue

            text = skill_file.read_text(encoding="utf-8")
            engine.add_chunk(
                text=text,
                store="procedural",
                source=str(skill_file.relative_to(Path.home())),
                source_type="skill",
                importance=0.8,
                metadata={"skill_name": skill_file.parent.name}
            )
            indexed += 1
        except Exception as e:
            print(f"[WARN] Failed to index skill {skill_file}: {e}")
    return indexed


def run_full_index(engine: MemoryEngine = None, since: datetime = None):
    """Run complete indexing pass."""
    own_engine = engine is None
    if own_engine:
        engine = MemoryEngine(DB_PATH)

    print("[INDEX] Starting full index...")
    total = 0

    n = index_memory_files(engine, since)
    print(f"[INDEX] Memory files: {n}")
    total += n

    n = index_workspace_docs(engine, since)
    print(f"[INDEX] Workspace docs: {n}")
    total += n

    n = index_skills(engine, since)
    print(f"[INDEX] Skills: {n}")
    total += n

    stats = engine.get_stats()
    print(f"[INDEX] Done. Total new: {total}. DB: {stats['total_chunks']} chunks, {stats['db_size_mb']} MB")

    if own_engine:
        engine.close()

    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Full re-index")
    parser.add_argument("--since", type=str, help="ISO datetime to index since")
    args = parser.parse_args()

    since = None
    if args.since:
        since = datetime.fromisoformat(args.since)

    run_full_index(since=since)
