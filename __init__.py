"""HAM v2 — Hermes Advanced Memory as a native MemoryProvider plugin.

Replaces the orphaned HAM v1 (venv `ham` package + ham.db) which was never
wired into the agent runtime. v2 plugs into the host's memory lifecycle:

  prefetch()        — hybrid recall of relevant facts, injected each turn
  sync_turn()       — buffers completed turns (no writes yet)
  on_session_end()  — one cheap LLM call: extract facts, reconcile
                      (add/update/supersede), store episode summary
  on_pre_compress() — same extraction before context compression discards turns
  on_memory_write() — mirrors built-in MEMORY.md/USER.md writes as facts

Storage: $HERMES_HOME/memory/ham_v2.db (SQLite, FTS5, numpy cosine — no
sqlite-vec, no external embedding APIs). See store.py for rationale.

Config (all optional), in config.yaml:

  memory:
    provider: ham
  plugins:
    ham:
      db_path: $HERMES_HOME/memory/ham_v2.db
      embed_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
      prefetch_top_k: 4
      prefetch_min_match: 0.50  # injection gate on query-relatedness (vec+bm25)
      extract_enabled: true
      extract_provider: ""      # empty = auxiliary 'compression' task model
      extract_model: ""
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.memory_provider import MemoryProvider

from .store import HamStore, DEFAULT_EMBED_MODEL
from . import extract as _extract

logger = logging.getLogger(__name__)

_PLUGIN_KEY = "ham"

# Prefetch query construction. Real usage is chat-style: 43% of measured user
# messages are under 60 chars and anaphoric ("kasuj", "a co z tamtym?") — the
# message alone is a useless retrieval key, but the previous turn names the
# topic. Long messages carry their own topic and are used as-is.
MIN_QUERY_CHARS = 15
SHORT_QUERY_CHARS = 80
CONTEXT_SNIPPET_CHARS = 300
MAX_QUERY_CHARS = 900

# How many recent turns' injected facts to suppress from re-injection.
DEDUP_TURNS = 3


def build_prefetch_query(message: str, prev_user: str = "", prev_asst: str = "") -> str:
    """Retrieval key for a turn: the message, topic-anchored when it's short.

    Returns "" when there is nothing searchable (trivial message, no context).
    """
    message = (message or "").strip()
    if len(message) >= SHORT_QUERY_CHARS:
        return message[:MAX_QUERY_CHARS]
    context = " ".join(
        s.strip()[:CONTEXT_SNIPPET_CHARS]
        for s in (prev_user, prev_asst) if s and s.strip()
    )
    query = (context + " " + message).strip() if context else message
    if len(query) < MIN_QUERY_CHARS:
        return ""
    return query[:MAX_QUERY_CHARS]

HAM_TOOL_SCHEMA = {
    "name": "ham_memory",
    "description": (
        "Long-term local memory (HAM). Relevant facts are auto-injected each "
        "turn as <memory-context>; use this tool for anything beyond that.\n"
        "• recall — search memory explicitly (before answering questions about "
        "the user's past decisions/preferences, check here first).\n"
        "• remember — store a durable fact NOW (user said 'zapamiętaj', or the "
        "always-injected MEMORY.md is full and rejected a write).\n"
        "• forget — mark a fact as no longer true (keeps history, use the id "
        "shown in recall results).\n"
        "• status — store statistics.\n"
        "Do NOT store task progress, secrets, or procedures (procedures → skills)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["recall", "remember", "forget", "status"]},
            "query": {"type": "string", "description": "Search query (recall)."},
            "text": {"type": "string", "description": "Fact text (remember)."},
            "kind": {"type": "string",
                     "enum": ["user_pref", "project", "infra", "decision", "note"],
                     "description": "Fact kind (remember)."},
            "importance": {"type": "number", "description": "0..1 (remember), default 0.6."},
            "fact_id": {"type": "integer", "description": "Fact id (forget)."},
            "reason": {"type": "string", "description": "Why it's no longer true (forget)."},
            "top_k": {"type": "integer", "description": "Max results (recall), default 6."},
            "include_superseded": {"type": "boolean",
                                    "description": "Include outdated facts (recall) — history questions."},
        },
        "required": ["action"],
    },
}


def _load_plugin_config() -> dict:
    try:
        from hermes_cli.config import load_config, cfg_get
        config = load_config() or {}
        return cfg_get(config, "plugins", _PLUGIN_KEY, default={}) or {}
    except Exception:
        return {}


class HamMemoryProvider(MemoryProvider):
    def __init__(self, config: Optional[dict] = None):
        self._config = config if config is not None else _load_plugin_config()
        self._store: Optional[HamStore] = None
        self._session_id = ""
        self._agent_context = "primary"
        self._turn_buffer: Dict[str, List[Tuple[str, str]]] = {}
        self._buffer_lock = threading.Lock()
        # sid -> (query, block, injected_ids, ts)
        self._prefetch_cache: Dict[str, Tuple[str, str, List[int], float]] = {}
        # sid -> deque of per-turn injected id sets; union is suppressed.
        self._injected_recent: Dict[str, deque] = {}
        self._extracted_sessions: set = set()

        self._top_k = int(self._config.get("prefetch_top_k", 4))
        # 0.50 from the labeled bench: max-recall plateau ends at 0.45, junk
        # on should-be-quiet turns halves between 0.45 and 0.50.
        self._min_match = float(self._config.get("prefetch_min_match", 0.50))
        self._extract_enabled = bool(self._config.get("extract_enabled", True))
        self._extract_provider = str(self._config.get("extract_provider", "") or "")
        self._extract_model = str(self._config.get("extract_model", "") or "")

    # -- identity / lifecycle -------------------------------------------------

    @property
    def name(self) -> str:
        return "ham"

    def is_available(self) -> bool:
        return True  # SQLite always works; embeddings are optional at runtime

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "db_path", "description": "SQLite database path",
             "default": "$HERMES_HOME/memory/ham_v2.db"},
            {"key": "embed_model", "description": "fastembed model (local)",
             "default": DEFAULT_EMBED_MODEL},
            {"key": "prefetch_top_k", "description": "Facts injected per turn", "default": "4"},
            {"key": "prefetch_min_match", "description": "Min query-relatedness (vec+bm25) to inject a fact", "default": "0.50"},
            {"key": "extract_enabled", "description": "LLM fact extraction at session end",
             "default": "true", "choices": ["true", "false"]},
            {"key": "extract_provider", "description": "Extraction LLM provider (empty = auxiliary compression model)", "default": ""},
            {"key": "extract_model", "description": "Extraction LLM model", "default": ""},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        try:
            import yaml
            config_path = Path(hermes_home) / "config.yaml"
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})[_PLUGIN_KEY] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.warning("HAM save_config failed: %s", e)

    def _resolve_db_path(self, hermes_home: str) -> Path:
        raw = str(self._config.get("db_path") or "")
        if raw:
            raw = raw.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
            return Path(raw).expanduser()
        return Path(hermes_home) / "memory" / "ham_v2.db"

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or str(Path.home() / ".hermes")
        self._session_id = session_id or ""
        self._agent_context = kwargs.get("agent_context", "primary")
        embed_model = str(self._config.get("embed_model") or DEFAULT_EMBED_MODEL)
        self._store = HamStore(self._resolve_db_path(hermes_home), embed_model=embed_model)
        # Warm the embedder off the critical path — first model load can take
        # seconds; prefetch degrades to BM25 until it's ready.
        threading.Thread(
            target=self._store.embedder.warm, daemon=True, name="ham-embed-warm"
        ).start()

    def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store = None

    def backup_paths(self) -> List[str]:
        return []  # DB lives inside HERMES_HOME; `hermes backup` already covers it

    # -- system prompt ---------------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            s = self._store.stats()
        except Exception:
            return ""
        return (
            f"# HAM Memory\n"
            f"Local long-term memory active ({s['active']} facts). Relevant facts are "
            f"auto-injected each turn as <memory-context>. For explicit lookups or "
            f"storing durable facts use the ham_memory tool. Facts are also extracted "
            f"automatically when the session ends."
        )

    # -- read path ---------------------------------------------------------------

    def _search_query_for(self, sid: str, message: str) -> str:
        prev_u = prev_a = ""
        with self._buffer_lock:
            buf = self._turn_buffer.get(sid)
            if buf:
                prev_u, prev_a = buf[-1]
        return build_prefetch_query(message, prev_u, prev_a)

    def _recent_injected(self, sid: str) -> Set[int]:
        turns = self._injected_recent.get(sid)
        return set().union(*turns) if turns else set()

    def _recall_block(self, sid: str, message: str) -> Tuple[str, List[int]]:
        if not self._store:
            return "", []
        query = self._search_query_for(sid, message)
        if not query:
            return "", []
        exclude = self._recent_injected(sid)
        results = [
            r for r in self._store.search(query, top_k=self._top_k + len(exclude))
            if r["match_score"] >= self._min_match and r["id"] not in exclude
        ][: self._top_k]
        if not results:
            return "", []
        lines = ["## HAM recall (long-term memory)"]
        for r in results:
            age = time.strftime("%Y-%m", time.localtime(r["updated_at"]))
            lines.append(f"- (#{r['id']}, {r['kind']}, {age}) {r['text']}")
        return "\n".join(lines), [r["id"] for r in results]

    def _record_injected(self, sid: str, ids: List[int]) -> None:
        # Facts already in the model's context from the last few turns are
        # not re-injected; a turn without recall still advances the window.
        self._injected_recent.setdefault(sid, deque(maxlen=DEDUP_TURNS)).append(set(ids))

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        sid = session_id or self._session_id
        cached = self._prefetch_cache.pop(sid, None)
        if cached and cached[0] == query and time.time() - cached[3] < 600:
            self._record_injected(sid, cached[2])
            return cached[1]
        try:
            block, ids = self._recall_block(sid, query)
            self._record_injected(sid, ids)
            return block
        except Exception as e:
            logger.debug("HAM prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # Called on the manager's background worker — compute now, cache for
        # the next prefetch() so the foreground path is a dict lookup.
        sid = session_id or self._session_id
        try:
            block, ids = self._recall_block(sid, query)
            self._prefetch_cache[sid] = (query, block, ids, time.time())
        except Exception as e:
            logger.debug("HAM queue_prefetch failed: %s", e)

    # -- write path -----------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None) -> None:
        if self._agent_context != "primary":
            return  # never learn user facts from cron/subagent transcripts
        sid = session_id or self._session_id
        with self._buffer_lock:
            buf = self._turn_buffer.setdefault(sid, [])
            buf.append(((user_content or "")[:2000], (assistant_content or "")[:2000]))
            if len(buf) > 200:
                del buf[: len(buf) - 200]

    def _run_extraction(self, sid: str, messages: Optional[List[Dict[str, Any]]] = None,
                        *, clear: bool = True) -> Dict[str, Any]:
        if not self._store or not self._extract_enabled or self._agent_context != "primary":
            return {"ran": False, "skipped_reason": "disabled or non-primary context"}
        with self._buffer_lock:
            turns = list(self._turn_buffer.get(sid, []))
            if clear:
                self._turn_buffer.pop(sid, None)
        if not turns and messages:
            turns = _extract.turns_from_messages(messages)
        if not turns:
            return {"ran": False, "skipped_reason": "no turns"}
        # Guard against double extraction (session end fires from multiple paths).
        marker = (sid, len(turns))
        if marker in self._extracted_sessions:
            return {"ran": False, "skipped_reason": "already extracted"}
        self._extracted_sessions.add(marker)
        caller = _extract.default_llm_caller(self._extract_provider, self._extract_model)
        return _extract.extract_and_reconcile(
            self._store, turns, session_id=sid, llm_caller=caller,
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self._run_extraction(self._session_id, messages)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        # Turns about to be compressed away are our last chance to learn from
        # them verbatim. Extract now; the compressor keeps its own summary.
        self._run_extraction(self._session_id, messages, clear=True)
        return ""

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs) -> None:
        old_sid = self._session_id
        if reset and old_sid:
            # /reset or /new — the old conversation is over; learn from it.
            threading.Thread(
                target=self._run_extraction, args=(old_sid,), daemon=True,
                name="ham-extract-reset",
            ).start()
        elif old_sid and old_sid != new_session_id:
            # Logical continuation (/branch, compression) — carry the buffer over.
            with self._buffer_lock:
                buf = self._turn_buffer.pop(old_sid, None)
                if buf:
                    self._turn_buffer.setdefault(new_session_id, []).extend(buf)
        if rewound:
            with self._buffer_lock:
                self._turn_buffer.pop(new_session_id, None)
            self._injected_recent.pop(new_session_id, None)
        if old_sid and old_sid != new_session_id:
            self._injected_recent.pop(old_sid, None)
        self._session_id = new_session_id

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in MEMORY.md/USER.md adds so curated facts are searchable."""
        if not self._store or action not in ("add", "replace") or not content:
            return
        try:
            kind = "user_pref" if target == "user" else "note"
            self._store.add_fact(
                content.strip(), kind=kind, importance=0.8, source="builtin",
                session_id=self._session_id,
            )
        except Exception as e:
            logger.debug("HAM on_memory_write mirror failed: %s", e)

    # -- tools ------------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [HAM_TOOL_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "ham_memory":
            return json.dumps({"error": f"unknown tool {tool_name}"})
        if not self._store:
            return json.dumps({"error": "HAM store not initialized"})
        try:
            action = args.get("action")
            if action == "recall":
                query = (args.get("query") or "").strip()
                if not query:
                    return json.dumps({"error": "recall requires 'query'"})
                results = self._store.search(
                    query,
                    top_k=int(args.get("top_k") or 6),
                    include_superseded=bool(args.get("include_superseded")),
                    include_episodes=True,
                )
                out = [
                    {"id": r["id"], "kind": r["kind"], "status": r["status"],
                     "text": r["text"], "score": r["score"],
                     "updated": time.strftime("%Y-%m-%d", time.localtime(r["updated_at"])),
                     **({"session_id": r["session_id"]} if r["kind"] == "episode" and r["session_id"] else {})}
                    for r in results
                ]
                hint = ("For full transcripts of an episode use session_search "
                        "with its session_id.") if any(r["kind"] == "episode" for r in results) else ""
                return json.dumps({"results": out, "count": len(out), "hint": hint},
                                  ensure_ascii=False)

            if action == "remember":
                text = (args.get("text") or "").strip()
                if not text:
                    return json.dumps({"error": "remember requires 'text'"})
                fid = self._store.add_fact(
                    text,
                    kind=args.get("kind") or "note",
                    importance=float(args.get("importance") or 0.6),
                    source="agent_tool",
                    session_id=self._session_id,
                )
                return json.dumps({"stored": True, "fact_id": fid})

            if action == "forget":
                fid = args.get("fact_id")
                if fid is None:
                    return json.dumps({"error": "forget requires 'fact_id'"})
                ok = self._store.invalidate(int(fid), reason=args.get("reason") or "")
                return json.dumps({"forgotten": ok, "fact_id": int(fid),
                                   "note": "kept as superseded history" if ok else "not found"})

            if action == "status":
                return json.dumps(self._store.stats(), ensure_ascii=False)

            return json.dumps({"error": f"unknown action {action}"})
        except Exception as e:
            logger.warning("ham_memory tool error: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})


def register(ctx) -> None:
    ctx.register_memory_provider(HamMemoryProvider())
