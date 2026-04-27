#!/usr/bin/env python3
"""
Hermes Advanced Memory (HAM) 🍖
Local multi-store memory with hybrid retrieval.

Stores:
- episodic: raw conversations/events (time-series)
- semantic: facts, concepts, user preferences (deduplicated knowledge)
- procedural: skills, workflows, how-to (actionable knowledge)
- archive: compressed summaries of old episodic memory

Retrieval: hybrid = BM25(FTS5) * 0.3 + vector_cosine * 0.4 + recency * 0.2 + importance * 0.1
"""

import json
import hashlib
import sqlite3
import sqlite_vec
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import struct
import os

# Load .env if present
_env_path = Path.home() / ".hermes" / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if line.strip() and not line.startswith('#') and '=' in line:
            key, val = line.split('=', 1)
            os.environ.setdefault(key.strip(), val.strip())

DB_PATH = Path.home() / ".hermes" / "memory" / "ham.db"
EMBEDDING_DIMS = 3072  # gemini-embedding-001
DEFAULT_CHUNK_SIZE = 400  # tokens approx
DEFAULT_CHUNK_OVERLAP = 80



class EmbeddingProjector:
    """Project embeddings of any dimension to target_dim via seeded random Gaussian."""
    _matrices: dict = {}

    @classmethod
    def project(cls, vec: List[float], target_dim: int = EMBEDDING_DIMS, seed: int = 42) -> List[float]:
        src_dim = len(vec)
        if src_dim == target_dim:
            return vec
        key = (src_dim, target_dim, seed)
        if key not in cls._matrices:
            rng = np.random.RandomState(seed)
            cls._matrices[key] = rng.randn(target_dim, src_dim) / np.sqrt(src_dim)
        proj = np.dot(cls._matrices[key], vec)
        norm = np.linalg.norm(proj)
        return (proj / norm).tolist() if norm > 0 else proj.tolist()


class GeminiProvider:
    """Primary: Gemini embedding-001, 3072 dims."""
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.max_retries = max_retries
        self.dims = 3072
        self.name = "gemini"

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("No GEMINI_API_KEY")
        import urllib.request, json, time
        results = []
        for text in texts:
            embedding = None
            for attempt in range(self.max_retries):
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"
                    data = json.dumps({"content": {"parts": [{"text": text}]}}).encode()
                    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        result = json.loads(resp.read().decode())
                        embedding = result["embedding"]["values"]
                        break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        time.sleep(2 ** attempt + 1)
                    else:
                        time.sleep(0.5 * (attempt + 1))
                except Exception:
                    time.sleep(0.5 * (attempt + 1))
            if embedding is None:
                raise RuntimeError(f"Gemini failed after {self.max_retries} attempts")
            results.append(embedding)
        return results


class OpenRouterProvider:
    """Fallback: OpenRouter with text-embedding-3-large (3072 dims, OpenAI compat)."""
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 2):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.max_retries = max_retries
        self.dims = 3072
        self.name = "openrouter"

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("No OPENROUTER_API_KEY")
        import openai, time
        client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        for attempt in range(self.max_retries):
            try:
                resp = client.embeddings.create(input=texts, model="openai/text-embedding-3-large")
                return [d.embedding for d in resp.data]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"OpenRouter failed: {e}")
        return []


class FastEmbedProvider:
    """Local fallback: fastembed (ONNX, no torch). Projects to target_dim."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)
        raw = list(self.model.embed(["test"]))[0]
        self.dims = len(raw)
        self.name = "fastembed"

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [list(vec) for vec in self.model.embed(texts)]


class HashProvider:
    """Last-resort deterministic fake embedding. Degrades vector search quality."""
    def __init__(self, dims: int = EMBEDDING_DIMS):
        self.dims = dims
        self.name = "hash"

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.normal(0, 1, self.dims)
            norm = np.linalg.norm(vec)
            vec = vec / norm if norm > 0 else vec
            results.append(vec.tolist())
        return results


class EmbeddingManager:
    """Orchestrates cache + provider fallback chain + projection."""
    def __init__(self, conn: sqlite3.Connection, target_dim: int = EMBEDDING_DIMS):
        self.conn = conn
        self.target_dim = target_dim
        self.providers = [
            GeminiProvider(),
            OpenRouterProvider(),
            FastEmbedProvider(),
            HashProvider(),
        ]

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _now_ts(self) -> int:
        return int(datetime.now().timestamp())

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], str]:
        """Return (embeddings, provider_name). Uses cache + fallback chain."""
        if not texts:
            return [], "none"

        text_hashes = [self._text_hash(t) for t in texts]
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []

        for i, h in enumerate(text_hashes):
            row = self.conn.execute(
                "SELECT embedding, model FROM embedding_cache WHERE text_hash = ?",
                (h,)
            ).fetchone()
            if row:
                count = len(row[0]) // 4
                embeddings[i] = list(struct.unpack(f'{count}f', row[0]))

        missing_indices = [i for i, e in enumerate(embeddings) if e is None]

        if not missing_indices:
            # All from cache — return the most common provider (default gemini if mixed)
            providers = [row[0] for row in self.conn.execute(
                "SELECT model FROM embedding_cache WHERE text_hash IN ({})".format(
                    ','.join('?' * len(text_hashes))
                ), text_hashes
            ).fetchall()]
            from collections import Counter
            provider = Counter(providers).most_common(1)[0][0] if providers else "gemini"
            return [e for e in embeddings if e is not None], provider

        missing_texts = [texts[i] for i in missing_indices]

        for provider in self.providers:
            try:
                raw_embeddings = provider.embed(missing_texts)
                for idx_in_missing, raw_emb in enumerate(raw_embeddings):
                    if len(raw_emb) != self.target_dim:
                        raw_emb = EmbeddingProjector.project(raw_emb, self.target_dim)
                    embeddings[missing_indices[idx_in_missing]] = raw_emb

                # Cache
                for idx_in_missing, h in enumerate([text_hashes[i] for i in missing_indices]):
                    emb = embeddings[missing_indices[idx_in_missing]]
                    blob = struct.pack(f'{len(emb)}f', *emb)
                    self.conn.execute(
                        "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, model, dims, created_at) VALUES (?, ?, ?, ?, ?)",
                        (h, blob, provider.name, len(emb), self._now_ts())
                    )
                self.conn.commit()
                return [e for e in embeddings if e is not None], provider.name
            except Exception as e:
                print(f"[EmbeddingManager] Provider {provider.name} failed: {e}")
                continue

        return [e for e in embeddings if e is not None], "failed"


@dataclass
class MemoryChunk:
    id: str
    store: str  # episodic | semantic | procedural | archive
    text: str
    embedding: Optional[List[float]]
    source: str  # file path or session_id
    source_type: str  # file | session | tool | cron
    created_at: datetime
    updated_at: datetime
    access_count: int
    last_accessed: Optional[datetime]
    importance: float  # 0.0-1.0, auto-calculated or manual
    recency_boost: float  # calculated at query time
    metadata: Dict

    def to_dict(self):
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        d['last_accessed'] = self.last_accessed.isoformat() if self.last_accessed else None
        d['embedding'] = None  # don't serialize embeddings to JSON
        return d


class MemoryEngine:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.embedding_mgr = EmbeddingManager(self.conn)

    def _init_db(self):
        """Create tables, run migrations, set up triggers."""
        # Enable sqlite-vec
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Schema version tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at INTEGER NOT NULL
            )
        """)

        current = self.conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        current_version = current[0] if current else 0

        if current_version < 1:
            self._migrate_v1()
            self.conn.execute(
                "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                (1, self._now_ts()),
            )

        self.conn.commit()

    def _migrate_v1(self):
        """Initial schema."""
        # Main chunks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                store TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                source_type TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed INTEGER,
                importance REAL DEFAULT 0.5,
                metadata TEXT DEFAULT '{}',
                UNIQUE(id)
            )
        """)

        # FTS5 for keyword/BM25 search
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, source, store,
                content='chunks',
                content_rowid='rowid'
            )
        """)

        # Vector virtual table
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                embedding float[{EMBEDDING_DIMS}]
            )
        """)

        # Embedding cache (hash -> embedding)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                dims INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)

        # Semantic links (graph relationships between chunks)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_links (
                from_chunk TEXT NOT NULL,
                to_chunk TEXT NOT NULL,
                relation TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (from_chunk, to_chunk, relation)
            )
        """)

        # Consolidation log
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                details TEXT,
                created_at INTEGER NOT NULL
            )
        """)

        # Indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_store ON chunks(store)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_links_from ON semantic_links(from_chunk)")

        # Triggers to keep FTS in sync
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text, source, store)
                VALUES (new.rowid, new.text, new.source, new.store);
            END
        """)
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, source, store)
                VALUES ('delete', old.rowid, old.text, old.source, old.store);
            END
        """)
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text, source, store)
                VALUES ('delete', old.rowid, old.text, old.source, old.store);
                INSERT INTO chunks_fts(rowid, text, source, store)
                VALUES (new.rowid, new.text, new.source, new.store);
            END
        """)

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _now_ts(self) -> int:
        return int(datetime.now().timestamp())

    def _chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                    overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
        """Simple token-aware chunking (approximate via words)."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - overlap if end < len(words) else end
        return chunks

    def add_chunk(self, text: str, store: str, source: str, source_type: str,
                  importance: float = 0.5, metadata: Dict = None,
                  chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Add text to memory, auto-chunk, embed (batched), index. Returns chunk IDs."""
        assert store in ('episodic', 'semantic', 'procedural', 'archive')
        metadata = metadata or {}
        chunks = self._chunk_text(text, chunk_size) if len(text.split()) > chunk_size * 1.5 else [text]
        chunk_ids = []
        now = self._now_ts()

        # Batch embed all chunks
        embeddings, provider = self.embedding_mgr.get_embeddings(chunks)
        if provider == "failed":
            raise RuntimeError("All embedding providers failed")

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{store}_{self._text_hash(chunk_text)}_{now}_{i}"
            embedding = embeddings[i]
            embedding_json = json.dumps(embedding)

            self.conn.execute(
                """INSERT INTO chunks (id, store, text, source, source_type, created_at, updated_at,
                    access_count, importance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (chunk_id, store, chunk_text, source, source_type, now, now,
                 importance, json.dumps(metadata))
            )
            self.conn.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES (last_insert_rowid(), ?)",
                (embedding_json,)
            )
            chunk_ids.append(chunk_id)

        self.conn.commit()
        return chunk_ids

    def hybrid_search(self, query: str, store: Optional[str] = None,
                      top_k: int = 10, recency_days: int = 30) -> List[Dict]:
        """
        Hybrid retrieval:
        score = 0.4 * cosine_sim + 0.3 * BM25_norm + 0.2 * recency + 0.1 * importance
        Falls back to BM25+recency only if embedding provider mismatches DB vectors.
        """
        query_embedding, q_provider = self.embedding_mgr.get_embeddings([query])
        query_embedding = query_embedding[0]
        query_json = json.dumps(query_embedding)
        now = self._now_ts()
        recency_cutoff = now - (recency_days * 86400)

        # Check if DB vectors are from the same provider
        has_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        can_vector = (q_provider == "gemini") or (has_chunks == 0)

        # Vector search via sqlite-vec (only if provider matches)
        vec_results = {}
        if can_vector:
            store_where = f"AND c.store = '{store}'" if store else ""
            for row in self.conn.execute(f"""
                SELECT c.id, c.rowid, vec_distance_cosine(c_vec.embedding, ?) as dist
                FROM chunks_vec AS c_vec
                JOIN chunks AS c ON c.rowid = c_vec.rowid
                WHERE 1=1 {store_where}
                ORDER BY dist
                LIMIT {top_k * 3}
            """, (query_json,)):
                vec_results[row[0]] = 1.0 - row[2]  # convert distance to similarity

        # BM25 search via FTS5
        bm25_results = {}
        # Build FTS5 query: individual words with AND
        words = [w for w in query.replace('"', '').split() if len(w) > 2]
        safe_query = ' AND '.join(words) if words else query.replace('"', '""')
        fts_store = f"AND fts.store = '{store}'" if store else ""
        for row in self.conn.execute(f"""
            SELECT c.id, rank
            FROM chunks_fts AS fts
            JOIN chunks AS c ON c.rowid = fts.rowid
            WHERE chunks_fts MATCH ? {fts_store}
            ORDER BY rank
            LIMIT {top_k * 3}
        """, (safe_query,)):
            # rank is negative BM25, higher = better match
            bm25_score = min(1.0, max(0.0, (-row[1]) / 10.0))  # normalize roughly
            bm25_results[row[0]] = bm25_score

        # Combine scores
        all_ids = set(vec_results.keys()) | set(bm25_results.keys())
        scored = []

        for chunk_id in all_ids:
            row = self.conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
            if not row:
                continue

            vec_score = vec_results.get(chunk_id, 0.0)
            bm25_score = bm25_results.get(chunk_id, 0.0)

            # Recency: exponential decay
            age_days = (now - row['created_at']) / 86400
            recency_score = np.exp(-age_days / recency_days) if recency_days > 0 else 0.5

            importance = row['importance']

            if can_vector:
                final_score = (0.4 * vec_score +
                               0.3 * bm25_score +
                               0.2 * recency_score +
                               0.1 * importance)
            else:
                final_score = (0.55 * bm25_score +
                               0.35 * recency_score +
                               0.1 * importance)

            # Boost accessed items slightly (feedback loop)
            if row['access_count'] > 0:
                final_score *= (1 + min(0.1, row['access_count'] * 0.01))

            scored.append({
                'id': chunk_id,
                'score': round(final_score, 4),
                'breakdown': {
                    'vector': round(vec_score, 3),
                    'bm25': round(bm25_score, 3),
                    'recency': round(recency_score, 3),
                    'importance': round(importance, 3)
                },
                'text': row['text'],
                'store': row['store'],
                'source': row['source'],
                'created_at': datetime.fromtimestamp(row['created_at']).isoformat(),
                'access_count': row['access_count'],
                'metadata': json.loads(row['metadata'] or '{}')
            })

        scored.sort(key=lambda x: x['score'], reverse=True)

        # Update access counts
        for item in scored[:top_k]:
            self.conn.execute(
                "UPDATE chunks SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, item['id'])
            )
        self.conn.commit()

        return scored[:top_k]

    def remember(self, text: str, store: str = "episodic", source: str = "manual",
                 source_type: str = "user", importance: float = 0.5,
                 metadata: Dict = None) -> List[str]:
        """High-level: remember something."""
        return self.add_chunk(text, store, source, source_type, importance, metadata)

    def recall(self, query: str, store: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """High-level: recall relevant memories."""
        return self.hybrid_search(query, store=store, top_k=top_k)

    def get_stats(self) -> Dict:
        c = self.conn.execute
        return {
            'total_chunks': c("SELECT COUNT(*) FROM chunks").fetchone()[0],
            'by_store': {row[0]: row[1] for row in c("SELECT store, COUNT(*) FROM chunks GROUP BY store")},
            'embedding_cache_size': c("SELECT COUNT(*) FROM embedding_cache").fetchone()[0],
            'semantic_links': c("SELECT COUNT(*) FROM semantic_links").fetchone()[0],
            'total_accesses': c("SELECT COALESCE(SUM(access_count), 0) FROM chunks").fetchone()[0],
            'db_size_mb': round(self.db_path.stat().st_size / (1024 * 1024), 2)
        }

    def link_chunks(self, from_id: str, to_id: str, relation: str, strength: float = 0.5):
        """Create semantic link between two chunks."""
        self.conn.execute(
            "INSERT OR REPLACE INTO semantic_links (from_chunk, to_chunk, relation, strength, created_at) VALUES (?, ?, ?, ?, ?)",
            (from_id, to_id, relation, strength, self._now_ts())
        )
        self.conn.commit()

    def consolidate_episodic(self, days: int = 7):
        """Compress old episodic memories into archive summaries."""
        cutoff = self._now_ts() - (days * 86400)
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE store = 'episodic' AND created_at < ? AND access_count < 3",
            (cutoff,)
        ).fetchall()

        if len(rows) < 3:
            return {'action': 'skip', 'reason': 'too few old episodes', 'count': len(rows)}

        # Group by source (session/file)
        by_source = {}
        for row in rows:
            by_source.setdefault(row['source'], []).append(row)

        consolidated = 0
        for source, episodes in by_source.items():
            if len(episodes) < 2:
                continue

            # Create summary text
            texts = [e['text'][:500] for e in episodes]
            summary_text = f"[Archive summary of {len(episodes)} episodes from {source}]\n" + "\n---\n".join(texts[:5])

            # Store as archive
            self.add_chunk(
                summary_text, store='archive', source=source,
                source_type='auto_consolidation',
                importance=0.4,
                metadata={'consolidated_count': len(episodes), 'original_ids': [e['id'] for e in episodes]}
            )

            # Delete old episodic chunks
            for ep in episodes:
                self.conn.execute("DELETE FROM chunks WHERE id = ?", (ep['id'],))
                self.conn.execute("DELETE FROM chunks_vec WHERE rowid NOT IN (SELECT rowid FROM chunks)")
            consolidated += len(episodes)

        self.conn.execute(
            "INSERT INTO consolidation_log (action, details, created_at) VALUES (?, ?, ?)",
            ('episodic_consolidation', f'Consolidated {consolidated} episodes', self._now_ts())
        )
        self.conn.commit()
        return {'action': 'consolidated', 'count': consolidated, 'sources': len(by_source)}

    def deduplicate_semantic(self, similarity_threshold: float = 0.92):
        """Find and merge duplicate semantic memories."""
        rows = self.conn.execute(
            "SELECT id, text FROM chunks WHERE store = 'semantic'"
        ).fetchall()

        merged = 0
        for i, row1 in enumerate(rows):
            for row2 in rows[i+1:]:
                # Simple text similarity (Jaccard on words)
                set1 = set(row1['text'].lower().split())
                set2 = set(row2['text'].lower().split())
                if not set1 or not set2:
                    continue
                jaccard = len(set1 & set2) / len(set1 | set2)

                if jaccard > similarity_threshold:
                    # Merge: keep more important/accessible one, append text
                    self.conn.execute(
                        """UPDATE chunks SET text = text || '\n[merged: ' || ? || ']',
                        importance = MAX(importance, (SELECT importance FROM chunks WHERE id = ?)),
                        access_count = access_count + (SELECT access_count FROM chunks WHERE id = ?),
                        updated_at = ? WHERE id = ?""",
                        (row2['text'][:200], row2['id'], row2['id'], self._now_ts(), row1['id'])
                    )
                    self.conn.execute("DELETE FROM chunks WHERE id = ?", (row2['id'],))
                    merged += 1

        self.conn.commit()
        return {'action': 'deduplicated', 'count': merged}

    def close(self):
        self.conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='HAM CLI')
    parser.add_argument('--remember', '-r', help='Text to remember')
    parser.add_argument('--store', '-s', default=None, choices=['episodic', 'semantic', 'procedural', 'archive'])
    parser.add_argument('--source', default='cli')
    parser.add_argument('--importance', type=float, default=0.5)
    parser.add_argument('--recall', '-q', help='Query to recall')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--consolidate', action='store_true')
    parser.add_argument('--db', type=str, default=str(DB_PATH))

    args = parser.parse_args()
    engine = MemoryEngine(Path(args.db))

    if args.remember:
        ids = engine.remember(args.remember, store=args.store, source=args.source, importance=args.importance)
        print(f"Remembered {len(ids)} chunk(s): {ids[0][:40]}...")

    elif args.recall:
        results = engine.recall(args.recall, store=args.store, top_k=args.top_k)
        print(f"\nFound {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [score: {r['score']}] [{r['store']}] {r['text'][:200]}")
            print(f"   source: {r['source']} | breakdown: {r['breakdown']}")
            print()

    elif args.consolidate:
        result = engine.consolidate_episodic()
        print(json.dumps(result, indent=2))

    elif args.stats:
        print(json.dumps(engine.get_stats(), indent=2))

    else:
        parser.print_help()

    engine.close()


if __name__ == "__main__":
    main()
