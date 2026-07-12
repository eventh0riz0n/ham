"""HAM v2 fact store — local SQLite with hybrid retrieval.

Design decisions (deliberate departures from HAM v1):

- Facts, not chunks. One row = one durable fact or one episode summary.
  No file indexing, no skill indexing — skills and workspace docs have
  their own discovery mechanisms; duplicating them here crowded fact
  recall in v1 (99% of the corpus was SKILL.md dumps).
- Single embedding space. One local fastembed model, recorded per row.
  v1 mixed three provider spaces, which made vector recall depend on
  which API key happened to work that day. If the embedder is
  unavailable, rows are stored with NULL embeddings and search degrades
  to BM25 — never fake/hash vectors.
- No sqlite-vec. At fact-store scale (thousands of rows) brute-force
  numpy cosine is <5 ms and avoids vec0's never-reclaimed chunk blobs
  (the mechanism that bloated v1 to 333 MB for 4.8 MB of text).
- Temporal supersede instead of delete. Contradicted facts get
  status='superseded' + superseded_by, so "what did I prefer before?"
  stays answerable and no extraction mistake is destructive.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

FACT_KINDS = ("user_pref", "project", "infra", "decision", "note", "episode")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'note',
    importance REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    invalidated_at INTEGER,
    last_accessed INTEGER,
    access_count INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'manual',
    session_id TEXT,
    embedding BLOB,
    emb_model TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(kind);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text,
    content='facts',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF text ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS maintenance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    details TEXT,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);
"""


class Embedder:
    """Lazy singleton wrapper around fastembed. Thread-safe.

    Returns None from embed() when fastembed or the model is unavailable —
    callers must treat embeddings as optional.
    """

    _instances: Dict[str, "Embedder"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    @classmethod
    def get(cls, model_name: str = DEFAULT_EMBED_MODEL) -> "Embedder":
        with cls._instances_lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = cls(model_name)
            return cls._instances[model_name]

    def warm(self) -> bool:
        """Load the model (downloads on first ever use). Returns success."""
        with self._lock:
            if self._model is not None:
                return True
            if self._failed:
                return False
            try:
                import warnings
                from fastembed import TextEmbedding
                with warnings.catch_warnings():
                    # fastembed warns that this model switched to mean pooling;
                    # v2 embeds everything with the same fastembed version, so
                    # the pooling change cannot desync query vs stored vectors.
                    warnings.simplefilter("ignore", UserWarning)
                    self._model = TextEmbedding(model_name=self.model_name)
                return True
            except Exception as e:
                logger.warning("HAM embedder unavailable (%s): %s", self.model_name, e)
                self._failed = True
                return False

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []
        if not self.warm():
            return None
        try:
            with self._lock:
                return [list(map(float, v)) for v in self._model.embed(texts)]
        except Exception as e:
            logger.warning("HAM embed failed: %s", e)
            return None


def _pack(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> List[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _fts_query(query: str) -> str:
    """Quote each word token so FTS5 never sees MATCH syntax."""
    words = re.findall(r"\w+", query, re.UNICODE)
    return " OR ".join(f'"{w}"' for w in words[:24])


class HamStore:
    def __init__(self, db_path: Path | str, embed_model: str = DEFAULT_EMBED_MODEL):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model
        self.embedder = Embedder.get(embed_model)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._write_lock = threading.Lock()
        self._init_schema()
        # In-memory vector matrix cache: (ids, numpy matrix). Invalidated on writes.
        self._vec_cache = None

    def _init_schema(self):
        with self._write_lock:
            self.conn.executescript(_SCHEMA)
            cur = self.conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0]
            if not cur:
                self.conn.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (1, ?)",
                    (int(time.time()),),
                )
            self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # -- writes ---------------------------------------------------------------

    def add_fact(
        self,
        text: str,
        *,
        kind: str = "note",
        importance: float = 0.5,
        source: str = "manual",
        session_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
        embed: bool = True,
    ) -> int:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty fact text")
        if kind not in FACT_KINDS:
            kind = "note"
        importance = max(0.0, min(1.0, float(importance)))
        now = int(time.time())

        # Exact-duplicate guard: same active text → touch, don't re-add.
        dup = self.conn.execute(
            "SELECT id FROM facts WHERE status='active' AND text = ? LIMIT 1", (text,)
        ).fetchone()
        if dup:
            with self._write_lock:
                self.conn.execute(
                    "UPDATE facts SET updated_at = ?, importance = MAX(importance, ?) WHERE id = ?",
                    (now, importance, dup["id"]),
                )
                self.conn.commit()
            return int(dup["id"])

        blob, model = None, None
        if embed:
            vecs = self.embedder.embed([text])
            if vecs:
                blob, model = _pack(vecs[0]), self.embed_model

        with self._write_lock:
            cur = self.conn.execute(
                """INSERT INTO facts(text, kind, importance, status, created_at, updated_at,
                                     source, session_id, embedding, emb_model, meta)
                   VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?)""",
                (text, kind, importance, now, now, source, session_id or None,
                 blob, model, json.dumps(meta or {}, ensure_ascii=False)),
            )
            self.conn.commit()
            self._vec_cache = None
            return int(cur.lastrowid)

    def update_fact(self, fact_id: int, new_text: str, *, importance: Optional[float] = None) -> bool:
        """Rewrite a fact in place (same identity, refined wording)."""
        new_text = (new_text or "").strip()
        row = self.conn.execute("SELECT id FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if not row or not new_text:
            return False
        blob, model = None, None
        vecs = self.embedder.embed([new_text])
        if vecs:
            blob, model = _pack(vecs[0]), self.embed_model
        now = int(time.time())
        with self._write_lock:
            if importance is not None:
                self.conn.execute(
                    "UPDATE facts SET text=?, embedding=?, emb_model=?, updated_at=?, importance=? WHERE id=?",
                    (new_text, blob, model, now, max(0.0, min(1.0, float(importance))), fact_id),
                )
            else:
                self.conn.execute(
                    "UPDATE facts SET text=?, embedding=?, emb_model=?, updated_at=? WHERE id=?",
                    (new_text, blob, model, now, fact_id),
                )
            self.conn.commit()
            self._vec_cache = None
        return True

    def supersede(self, old_id: int, new_text: str, *, kind: Optional[str] = None,
                  importance: Optional[float] = None, source: str = "extraction",
                  session_id: str = "") -> Optional[int]:
        """Replace a fact with a newer truth; the old row stays queryable."""
        old = self.conn.execute("SELECT * FROM facts WHERE id = ?", (old_id,)).fetchone()
        if not old:
            return None
        new_id = self.add_fact(
            new_text,
            kind=kind or old["kind"],
            importance=importance if importance is not None else old["importance"],
            source=source,
            session_id=session_id,
            meta={"supersedes": old_id},
        )
        now = int(time.time())
        with self._write_lock:
            self.conn.execute(
                "UPDATE facts SET status='superseded', superseded_by=?, invalidated_at=? WHERE id=?",
                (new_id, now, old_id),
            )
            self.conn.commit()
            self._vec_cache = None
        return new_id

    def invalidate(self, fact_id: int, reason: str = "") -> bool:
        """Mark a fact as no longer true, without a replacement."""
        row = self.conn.execute("SELECT id, meta FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if not row:
            return False
        meta = json.loads(row["meta"] or "{}")
        if reason:
            meta["invalidate_reason"] = reason
        now = int(time.time())
        with self._write_lock:
            self.conn.execute(
                "UPDATE facts SET status='superseded', invalidated_at=?, meta=? WHERE id=?",
                (now, json.dumps(meta, ensure_ascii=False), fact_id),
            )
            self.conn.commit()
            self._vec_cache = None
        return True

    # -- reads ----------------------------------------------------------------

    def get(self, fact_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return self._row_dict(row) if row else None

    def _row_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = {k: row[k] for k in row.keys() if k != "embedding"}
        d["meta"] = json.loads(d.get("meta") or "{}")
        return d

    def _vec_matrix(self):
        """(ids, matrix) over active facts with embeddings from the current model."""
        if self._vec_cache is not None:
            return self._vec_cache
        import numpy as np
        rows = self.conn.execute(
            "SELECT id, embedding FROM facts WHERE status='active' AND embedding IS NOT NULL AND emb_model = ?",
            (self.embed_model,),
        ).fetchall()
        if not rows:
            self._vec_cache = ([], None)
            return self._vec_cache
        ids = [r["id"] for r in rows]
        mat = np.array([_unpack(r["embedding"]) for r in rows], dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vec_cache = (ids, mat / norms)
        return self._vec_cache

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        kinds: Optional[List[str]] = None,
        include_episodes: bool = False,
        include_superseded: bool = False,
        recency_half_life_days: float = 90.0,
        touch: bool = True,
    ) -> List[Dict[str, Any]]:
        """Hybrid recall ranked by Reciprocal Rank Fusion of the two lanes.

        RRF (k=60) replaced the v2.0 weighted sum after benchmarking on real
        turns: rank positions are comparable across lanes, raw cosine and
        normalized-BM25 magnitudes are not (recall@4 0.19 → 0.27 on the
        labeled bench). The legacy weighted score is still computed — it
        breaks RRF ties and stays visible in results; `match_score` is the
        query-relatedness part only, meant for injection gating.
        """
        query = (query or "").strip()
        if not query:
            return []
        top_k = max(1, min(int(top_k), 50))
        now = int(time.time())

        # Vector lane
        vec_scores: Dict[int, float] = {}
        vec_ranks: Dict[int, int] = {}
        qvecs = self.embedder.embed([query])
        if qvecs:
            import numpy as np
            ids, mat = self._vec_matrix()
            if mat is not None and len(ids):
                q = np.array(qvecs[0], dtype="float32")
                qn = np.linalg.norm(q)
                if qn > 0:
                    sims = mat @ (q / qn)
                    order = np.argsort(-sims)[: top_k * 4]
                    for rank, i in enumerate(order):
                        vec_scores[ids[int(i)]] = float(sims[int(i)])
                        vec_ranks[ids[int(i)]] = rank

        # BM25 lane
        bm25_scores: Dict[int, float] = {}
        bm25_ranks: Dict[int, int] = {}
        fq = _fts_query(query)
        if fq:
            try:
                for rank, row in enumerate(self.conn.execute(
                    "SELECT rowid, rank FROM facts_fts WHERE facts_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fq, top_k * 4),
                )):
                    bm25_scores[row["rowid"]] = min(1.0, max(0.0, (-row["rank"]) / 12.0))
                    bm25_ranks[row["rowid"]] = rank
            except sqlite3.OperationalError:
                pass

        candidates = set(vec_scores) | set(bm25_scores)
        if not candidates:
            return []

        placeholders = ",".join("?" * len(candidates))
        rows = {
            r["id"]: r
            for r in self.conn.execute(
                f"SELECT * FROM facts WHERE id IN ({placeholders})", list(candidates)
            )
        }

        import math
        scored = []
        for fid in candidates:
            row = rows.get(fid)
            if row is None:
                continue
            if not include_superseded and row["status"] != "active":
                continue
            if not include_episodes and row["kind"] == "episode":
                continue
            if kinds and row["kind"] not in kinds:
                continue
            vec = vec_scores.get(fid)
            bm = bm25_scores.get(fid, 0.0)
            age_days = max(0.0, (now - row["updated_at"]) / 86400.0)
            rec = math.exp(-age_days * math.log(2) / max(recency_half_life_days, 1.0))
            imp = float(row["importance"])
            if vec is not None:
                score = 0.5 * max(0.0, vec) + 0.3 * bm + 0.1 * rec + 0.1 * imp
                match = (0.5 * max(0.0, vec) + 0.3 * bm) / 0.8
            else:
                score = 0.6 * bm + 0.2 * rec + 0.2 * imp
                match = bm
            rrf = 0.0
            if fid in vec_ranks:
                rrf += 1.0 / (60 + vec_ranks[fid] + 1)
            if fid in bm25_ranks:
                rrf += 1.0 / (60 + bm25_ranks[fid] + 1)
            d = self._row_dict(row)
            d["score"] = round(score, 4)
            # Query-relatedness alone (vec+bm25, no recency/importance). Gate on
            # this, not on score: a fresh important fact scores ~0.2 before the
            # text matches anything, which let junk through score thresholds.
            d["match_score"] = round(match, 4)
            d["rrf"] = round(rrf, 5)
            d["score_parts"] = {
                "vector": round(vec, 3) if vec is not None else None,
                "bm25": round(bm, 3),
                "recency": round(rec, 3),
                "importance": round(imp, 3),
            }
            scored.append(d)

        scored.sort(key=lambda x: (x["rrf"], x["score"]), reverse=True)
        result = scored[:top_k]

        if touch and result:
            with self._write_lock:
                for item in result:
                    self.conn.execute(
                        "UPDATE facts SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                        (now, item["id"]),
                    )
                self.conn.commit()
        return result

    def list_facts(self, *, status: str = "active", kind: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        where = ["status = ?"] if status != "all" else []
        params: List[Any] = [status] if status != "all" else []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        params.extend([limit, max(0, int(offset))])
        rows = self.conn.execute(
            f"SELECT * FROM facts {clause} ORDER BY updated_at DESC LIMIT ? OFFSET ?", params
        ).fetchall()
        return [self._row_dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        c = self.conn.execute
        return {
            "db_path": str(self.db_path),
            "total": c("SELECT COUNT(*) FROM facts").fetchone()[0],
            "active": c("SELECT COUNT(*) FROM facts WHERE status='active'").fetchone()[0],
            "superseded": c("SELECT COUNT(*) FROM facts WHERE status='superseded'").fetchone()[0],
            "by_kind": {r[0]: r[1] for r in c(
                "SELECT kind, COUNT(*) FROM facts WHERE status='active' GROUP BY kind")},
            "embedded": c(
                "SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL AND emb_model = ?",
                (self.embed_model,),
            ).fetchone()[0],
            "embed_model": self.embed_model,
            "db_size_mb": round(self.db_path.stat().st_size / 1048576, 2) if self.db_path.exists() else 0,
        }

    # -- maintenance ------------------------------------------------------------

    def consolidate(self, *, sim_threshold: float = 0.93, stale_days: int = 60,
                    prune_superseded_days: int = 180) -> Dict[str, Any]:
        """Deterministic weekly hygiene. No LLM.

        1. Near-duplicate active facts (cosine > threshold, same kind) →
           keep the more-recently-updated / more-accessed one, supersede the other.
        2. Decay importance of facts not accessed in `stale_days` (floor 0.1).
        3. Hard-delete superseded rows older than `prune_superseded_days`.
        """
        import numpy as np
        now = int(time.time())
        merged = 0

        ids, mat = self._vec_matrix()
        if mat is not None and len(ids) > 1:
            rows = {r["id"]: r for r in self.conn.execute(
                f"SELECT id, kind, updated_at, access_count FROM facts WHERE id IN ({','.join('?'*len(ids))})",
                ids,
            )}
            sims = mat @ mat.T
            dead: set = set()
            for i in range(len(ids)):
                if ids[i] in dead:
                    continue
                for j in range(i + 1, len(ids)):
                    if ids[j] in dead or sims[i, j] < sim_threshold:
                        continue
                    a, b = rows[ids[i]], rows[ids[j]]
                    if a["kind"] != b["kind"] or a["kind"] == "episode":
                        continue
                    keep, drop = (a, b) if (
                        (a["access_count"], a["updated_at"]) >= (b["access_count"], b["updated_at"])
                    ) else (b, a)
                    with self._write_lock:
                        self.conn.execute(
                            "UPDATE facts SET status='superseded', superseded_by=?, invalidated_at=? WHERE id=?",
                            (keep["id"], now, drop["id"]),
                        )
                        self.conn.commit()
                    dead.add(drop["id"])
                    merged += 1
            if dead:
                self._vec_cache = None

        cutoff = now - stale_days * 86400
        with self._write_lock:
            decayed = self.conn.execute(
                """UPDATE facts SET importance = MAX(0.1, importance * 0.9)
                   WHERE status='active' AND kind != 'episode'
                     AND COALESCE(last_accessed, created_at) < ? AND importance > 0.1""",
                (cutoff,),
            ).rowcount
            prune_cutoff = now - prune_superseded_days * 86400
            pruned = self.conn.execute(
                "DELETE FROM facts WHERE status='superseded' AND COALESCE(invalidated_at, updated_at) < ?",
                (prune_cutoff,),
            ).rowcount
            self.conn.execute(
                "INSERT INTO maintenance_log(action, details, created_at) VALUES (?,?,?)",
                ("consolidate", json.dumps({"merged": merged, "decayed": decayed, "pruned": pruned}), now),
            )
            self.conn.commit()
            self._vec_cache = None
        try:
            self.conn.execute("VACUUM")
        except Exception:
            pass
        return {"merged": merged, "decayed": decayed, "pruned": pruned}

    def reembed_missing(self, batch: int = 64) -> int:
        """Embed rows with NULL/foreign-model embeddings. Returns count."""
        rows = self.conn.execute(
            "SELECT id, text FROM facts WHERE embedding IS NULL OR emb_model != ?",
            (self.embed_model,),
        ).fetchall()
        done = 0
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            vecs = self.embedder.embed([r["text"] for r in chunk])
            if not vecs:
                break
            with self._write_lock:
                for r, v in zip(chunk, vecs):
                    self.conn.execute(
                        "UPDATE facts SET embedding=?, emb_model=? WHERE id=?",
                        (_pack(v), self.embed_model, r["id"]),
                    )
                self.conn.commit()
            done += len(chunk)
        if done:
            self._vec_cache = None
        return done
