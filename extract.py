"""HAM v2 write path — end-of-session fact extraction and reconciliation.

One LLM call per session boundary (session end, pre-compress, /reset):
the model sees the conversation digest plus the nearest existing facts
and emits explicit operations (add / update / supersede / noop), Mem0-style.
Reconciling against existing facts at write time is what keeps the store
deduplicated and contradiction-free — retrieval never has to arbitrate
between five stale variants of the same fact.

All operations are non-destructive: supersede keeps the old row queryable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .store import HamStore, FACT_KINDS

logger = logging.getLogger(__name__)

MAX_DIGEST_CHARS = 14000
MAX_OPS = 10
MAX_FACT_CHARS = 400
MIN_TURNS = 2

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "update", "supersede", "noop"]},
                    "id": {"type": ["integer", "null"]},
                    "text": {"type": ["string", "null"]},
                    "kind": {"type": ["string", "null"],
                             "enum": list(FACT_KINDS) + [None]},
                    "importance": {"type": ["number", "null"]},
                },
                "required": ["op"],
            },
        },
        "session_summary": {"type": ["string", "null"]},
    },
    "required": ["operations"],
}

_INSTRUCTIONS = """You maintain the long-term memory of a personal AI agent (Hermes) for its user Beniamin.
Below you get (A) existing memory facts with numeric ids, and (B) a digest of the conversation that just ended (Polish and English mixed).

Decide which durable facts to write. Output JSON only.

WHAT COUNTS AS A DURABLE FACT:
- stable user preferences, conventions, decisions ("user prefers X", "we decided Y")
- lasting configuration/infrastructure facts (paths, ports, service names, model policy)
- corrections of things the agent got wrong and the user fixed

WHAT MUST NOT BE STORED:
- task progress, one-off errors, transient state, session narration
- anything already covered by an existing fact (emit noop or nothing)
- secrets, tokens, passwords, API keys
- procedures/how-tos (those belong in skills, not memory)

OPERATIONS:
- {"op":"add","text":...,"kind":...,"importance":0..1} — genuinely new fact
- {"op":"update","id":N,"text":...} — same fact, better/refined wording (typo fix, added detail)
- {"op":"supersede","id":N,"text":...} — the conversation CONTRADICTS fact N; text is the new truth
- {"op":"noop"} — nothing to change (valid output: {"operations":[{"op":"noop"}]})

kinds: user_pref (about the user), project (project state/decisions), infra (machines/services/paths), decision (agreed choices), note (other).
Write fact text in the language it naturally occurred in (Polish is fine). Max %d chars per fact. At most %d operations — fewer, well-chosen facts beat many weak ones. Most sessions yield 0-3 facts.

Also produce "session_summary": 1-3 sentences (what was worked on, what was decided), or null if the session was trivial small talk.
""" % (MAX_FACT_CHARS, MAX_OPS)


def build_digest(turns: List[Tuple[str, str]], max_chars: int = MAX_DIGEST_CHARS) -> str:
    """Flatten buffered (user, assistant) turns into a bounded digest.

    Keeps the head and tail of the session when over budget — openings carry
    intent, endings carry conclusions; the middle is usually tool churn.
    """
    parts = []
    for user, assistant in turns:
        u = (user or "").strip()
        a = (assistant or "").strip()
        if u:
            parts.append(f"USER: {u[:2000]}")
        if a:
            parts.append(f"ASSISTANT: {a[:2000]}")
    digest = "\n".join(parts)
    if len(digest) <= max_chars:
        return digest
    head = digest[: max_chars // 3]
    tail = digest[-(max_chars - len(head) - 30):]
    return head + "\n[... skrócono ...]\n" + tail


def turns_from_messages(messages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Fallback when no turn buffer exists: pair user/assistant text messages."""
    turns: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant":
            turns.append((pending_user or "", content))
            pending_user = None
    if pending_user:
        turns.append((pending_user, ""))
    return turns


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    # Tolerate leading prose before the JSON object.
    start = text.find("{")
    if start > 0:
        text = text[start:]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def default_llm_caller(provider: str = "", model: str = "") -> Callable[[str, str], str]:
    """Build a caller backed by the host's auxiliary LLM client.

    Uses the 'compression' auxiliary task config by default (already the
    user's cheap-model slot); plugin config may pin provider/model.
    Imported lazily — this module must stay importable outside the agent.
    """
    def _call(instructions: str, payload: str) -> str:
        from agent.auxiliary_client import call_llm
        response = call_llm(
            task=None if provider else "compression",
            provider=provider or None,
            model=model or None,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": payload},
            ],
            temperature=0.1,
            max_tokens=1400,
            timeout=90,
            extra_body={"response_format": {"type": "json_object"}},
        )
        try:
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "")
                    for p in content
                )
            return content or ""
        except (AttributeError, IndexError):
            return ""
    return _call


_ALIAS_INSTRUCTIONS = """You generate SEARCH ALIASES for facts stored in a personal AI agent's long-term memory (user: Beniamin, Polish/English mixed).

The problem you solve: conversations refer to these facts with different words than the fact text uses ("zabij meta observera" must find a fact written as "Metaobserver v2 zaimplementowany..."). For each fact below, output 3-6 short aliases:
- synonyms and conversational phrasings a user would actually type
- the other language's terms (Polish fact -> add English terms, and vice versa)
- entity/name variants (metaobserver / meta observer / obserwator)
- related action words ("wyłączyć", "kasuj", "disable" for facts about stopped services)

Do NOT restate the fact, do NOT invent new information, no full sentences — short search terms only. Output JSON only:
{"aliases": {"<fact_id>": ["term", "term", ...], ...}}"""

MAX_ALIASES_PER_FACT = 8
ALIAS_BATCH = 25


def expand_aliases(
    store: HamStore,
    fact_ids: Optional[List[int]] = None,
    *,
    llm_caller: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """Doc2query-style expansion: attach LLM-generated search aliases.

    With `fact_ids` expands exactly those facts; without, backfills active
    facts that have no aliases yet. Returns a report dict (never raises).
    """
    report = {"expanded": 0, "failed_batches": 0}
    try:
        if fact_ids:
            rows = [f for fid in fact_ids if (f := store.get(int(fid)))]
            targets = [{"id": f["id"], "text": f["text"]} for f in rows
                       if f["kind"] != "episode"]
        else:
            targets = store.facts_missing_aliases()
        if not targets:
            return report
        caller = llm_caller or default_llm_caller()
        for i in range(0, len(targets), ALIAS_BATCH):
            chunk = targets[i:i + ALIAS_BATCH]
            payload = "\n".join(f"[{t['id']}] {t['text']}" for t in chunk)
            parsed = _parse_json(caller(_ALIAS_INSTRUCTIONS, payload))
            aliases = (parsed or {}).get("aliases")
            if not isinstance(aliases, dict):
                report["failed_batches"] += 1
                logger.warning("HAM alias expansion: unparseable batch output")
                continue
            valid_ids = {t["id"] for t in chunk}
            for key, terms in aliases.items():
                try:
                    fid = int(key)
                except (TypeError, ValueError):
                    continue
                if fid not in valid_ids or not isinstance(terms, list):
                    continue
                terms = [str(t)[:60] for t in terms[:MAX_ALIASES_PER_FACT]]
                if store.set_aliases(fid, terms):
                    report["expanded"] += 1
        logger.info("HAM alias expansion: expanded=%d failed_batches=%d",
                    report["expanded"], report["failed_batches"])
        return report
    except Exception as e:
        logger.warning("HAM alias expansion failed (non-fatal): %s", e, exc_info=True)
        report["error"] = str(e)
        return report


def extract_and_reconcile(
    store: HamStore,
    turns: List[Tuple[str, str]],
    *,
    session_id: str = "",
    llm_caller: Optional[Callable[[str, str], str]] = None,
    candidate_facts: int = 20,
    expand_new: bool = False,
) -> Dict[str, Any]:
    """Run one extraction pass. Returns a report dict (never raises)."""
    report = {"ran": False, "ops_applied": 0, "added": [], "updated": [],
              "superseded": [], "episode": None, "skipped_reason": ""}
    try:
        if len(turns) < MIN_TURNS:
            report["skipped_reason"] = f"too few turns ({len(turns)})"
            return report

        digest = build_digest(turns)
        if len(digest) < 200:
            report["skipped_reason"] = "digest too short"
            return report

        # Candidate facts: what the store already believes about these topics.
        candidates = store.search(
            digest[:1500], top_k=candidate_facts, touch=False,
            include_episodes=False,
        )
        cand_block = "\n".join(
            f"[{c['id']}] ({c['kind']}) {c['text']}" for c in candidates
        ) or "(no existing facts matched)"
        known_ids = {c["id"] for c in candidates}

        payload = (
            "(A) EXISTING FACTS:\n" + cand_block +
            "\n\n(B) CONVERSATION DIGEST:\n" + digest
        )

        caller = llm_caller or default_llm_caller()
        raw = caller(_INSTRUCTIONS, payload)
        parsed = _parse_json(raw)
        if not parsed:
            report["skipped_reason"] = "unparseable LLM output"
            logger.warning("HAM extraction: unparseable output: %.200s", raw)
            return report

        report["ran"] = True
        ops = parsed.get("operations") or []
        for op in ops[:MAX_OPS]:
            if not isinstance(op, dict):
                continue
            action = op.get("op")
            text = (op.get("text") or "").strip()[:MAX_FACT_CHARS]
            kind = op.get("kind") if op.get("kind") in FACT_KINDS else "note"
            try:
                imp = max(0.0, min(1.0, float(op.get("importance") or 0.6)))
            except (TypeError, ValueError):
                imp = 0.6

            if action == "add" and text:
                fid = store.add_fact(text, kind=kind, importance=imp,
                                     source="extraction", session_id=session_id)
                report["added"].append(fid)
                report["ops_applied"] += 1
            elif action == "update" and text and op.get("id") in known_ids:
                if store.update_fact(int(op["id"]), text):
                    report["updated"].append(int(op["id"]))
                    report["ops_applied"] += 1
            elif action == "supersede" and text and op.get("id") in known_ids:
                nid = store.supersede(int(op["id"]), text, kind=kind,
                                      importance=imp, source="extraction",
                                      session_id=session_id)
                if nid:
                    report["superseded"].append({"old": int(op["id"]), "new": nid})
                    report["ops_applied"] += 1
            # noop / unknown ids / empty text → ignored by design

        # Opt-in (plugins.ham.alias_expansion): benchmarked neutral at current
        # store size — windowed prefetch queries already supply the topic
        # vocabulary aliases would add. Re-evaluate with a stronger embedder.
        if expand_new:
            new_ids = report["added"] + report["updated"] + \
                [s["new"] for s in report["superseded"]]
            if new_ids:
                expand_aliases(store, new_ids, llm_caller=caller)

        summary = (parsed.get("session_summary") or "").strip()
        if summary and len(summary) > 20:
            eid = store.add_fact(
                summary[:600], kind="episode", importance=0.4,
                source="session_summary", session_id=session_id,
                meta={"deep_recall": "session_search", "session_id": session_id},
            )
            report["episode"] = eid

        logger.info(
            "HAM extraction: session=%s ops=%d added=%d updated=%d superseded=%d episode=%s",
            session_id, report["ops_applied"], len(report["added"]),
            len(report["updated"]), len(report["superseded"]), report["episode"],
        )
        return report
    except Exception as e:
        logger.warning("HAM extraction failed (non-fatal): %s", e, exc_info=True)
        report["skipped_reason"] = f"error: {e}"
        return report
