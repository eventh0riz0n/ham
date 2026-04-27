#!/usr/bin/env python3
"""
Session hook for HAM.
Call this after each conversation to auto-save episodic memory.
Also retrieves relevant context before next conversation.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from memory_engine import MemoryEngine, DB_PATH


def save_session(session_text: str, session_id: str = None, metadata: dict = None):
    """Save a conversation session as episodic memory."""
    engine = MemoryEngine(DB_PATH)
    session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata = metadata or {}

    # Extract key facts automatically (simple heuristic)
    facts = extract_facts(session_text)
    if facts:
        for fact in facts:
            engine.remember(
                text=fact,
                store="semantic",
                source=session_id,
                source_type="auto_extracted",
                importance=0.6,
                metadata={"auto_extracted": True, "session": session_id}
            )

    # Save full session as episodic
    engine.remember(
        text=session_text,
        store="episodic",
        source=session_id,
        source_type="session",
        importance=0.5,
        metadata=metadata
    )

    stats = engine.get_stats()
    engine.close()
    return {"session_id": session_id, "facts_extracted": len(facts), **stats}


def extract_facts(text: str) -> list:
    """Simple fact extraction from session text."""
    facts = []
    lines = text.split('\n')

    # Look for declarative sentences that seem like facts/preferences/decisions
    for line in lines:
        line = line.strip()
        if len(line) < 20 or len(line) > 300:
            continue

        # Patterns that indicate facts/preferences
        patterns = [
            r"[Uu]ser (?:prefers?|likes?|wants?|hates?|dislikes?|loves?)",
            r"[Ww]e decided(?: to)?",
            r"[Ff]inal decision",
            r"[Cc]onfigured",
            r"[Ss]et up",
            r"[Nn]ew (?:skill|cron|job|config)",
            r"[Mm]igrated",
            r"[Bb]uilt",
            r"[Zz]mieniono",
            r"[Uu]stawiono",
            r"[Pp]referuje",
            r"[Nn]ie lubi",
        ]
        if any(re.search(p, line) for p in patterns):
            facts.append(line)

    return facts[:10]  # max 10 facts per session


def get_context_for_prompt(query: str, top_k: int = 5) -> str:
    """Retrieve relevant memory context to prepend to a prompt."""
    engine = MemoryEngine(DB_PATH)
    results = engine.recall(query, top_k=top_k)

    if not results:
        engine.close()
        return ""

    lines = ["[Relevant context from memory:]"]
    for r in results:
        store_emoji = {"episodic": "📝", "semantic": "🧠", "procedural": "⚙️", "archive": "📦"}.get(r['store'], "•")
        lines.append(f"{store_emoji} {r['text'][:300]} (score: {r['score']}, source: {r['source']})")

    engine.close()
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Path to session text file to save")
    parser.add_argument("--context", help="Query to get context for")
    parser.add_argument("--session-id", help="Session ID")
    args = parser.parse_args()

    if args.save:
        text = Path(args.save).read_text()
        result = save_session(text, args.session_id)
        print(json.dumps(result, indent=2))
    elif args.context:
        print(get_context_for_prompt(args.context))
    else:
        parser.print_help()
