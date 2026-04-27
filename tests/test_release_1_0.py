import sqlite3

from ham.memory_engine import MemoryEngine
from ham import indexer


class FixedProvider:
    def __init__(self, name, vectors):
        self.name = name
        self.vectors = vectors
        self.calls = 0

    def embed(self, texts):
        out = []
        for _ in texts:
            idx = min(self.calls, len(self.vectors) - 1)
            out.append(self.vectors[idx])
            self.calls += 1
        return out


def unit_vec(pos, dims=3072):
    vec = [0.0] * dims
    vec[pos] = 1.0
    return vec


def test_chunks_record_embedding_provider_and_vector_search_uses_matching_provider_only(tmp_db):
    engine = MemoryEngine(tmp_db)
    try:
        # Store two memories with intentionally different embedding providers.
        engine.embedding_mgr.providers = [FixedProvider("hash", [unit_vec(0)])]
        hash_id = engine.remember("hash provider apple", store="semantic", importance=0.1)[0]

        engine.embedding_mgr.providers = [FixedProvider("gemini", [unit_vec(1)])]
        gemini_id = engine.remember("gemini provider banana", store="semantic", importance=0.1)[0]

        rows = engine.conn.execute(
            "SELECT id, embedding_provider FROM chunks ORDER BY id"
        ).fetchall()
        providers = {row["id"]: row["embedding_provider"] for row in rows}
        assert providers[hash_id] == "hash"
        assert providers[gemini_id] == "gemini"

        # Query uses gemini vector matching the gemini chunk. Hash-provider chunks
        # must not receive vector scores in this query.
        engine.embedding_mgr.providers = [FixedProvider("gemini", [unit_vec(1)])]
        results = engine.recall("banana", top_k=5)

        by_id = {r["id"]: r for r in results}
        assert hash_id not in by_id or by_id[hash_id]["breakdown"]["vector"] == 0
        assert by_id[gemini_id]["breakdown"]["vector"] > 0
        assert results[0]["id"] == gemini_id
    finally:
        engine.close()


def test_existing_v1_database_migrates_to_provider_column(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version(version, applied_at) VALUES (1, 1)")
    conn.execute(
        """CREATE TABLE chunks (
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
        )"""
    )
    conn.execute(
        "INSERT INTO chunks(id, store, text, source, source_type, created_at, updated_at) VALUES ('c1', 'semantic', 'old text', 'test', 'manual', 1, 1)"
    )
    conn.commit()
    conn.close()

    engine = MemoryEngine(db)
    try:
        version = engine.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        row = engine.conn.execute("SELECT embedding_provider FROM chunks WHERE id = 'c1'").fetchone()
        assert version >= 2
        assert row[0] == "unknown"
    finally:
        engine.close()


def test_engine_management_methods(engine, tmp_path):
    chunk_id = engine.remember("delete me later", store="semantic", importance=0.9)[0]
    listed = engine.list_chunks(store="semantic", limit=10)
    assert any(item["id"] == chunk_id for item in listed)
    assert engine.get_chunk(chunk_id)["text"] == "delete me later"

    backup = tmp_path / "backup.db"
    assert engine.backup(backup) == backup
    assert backup.exists()
    assert engine.backup(backup, overwrite=True) == backup

    assert engine.delete_chunk(chunk_id) is True
    assert engine.get_chunk(chunk_id) is None


def test_doctor_reports_core_health(engine):
    report = engine.doctor()
    assert report["schema_version"] >= 2
    assert report["fts"] == "ok"
    assert report["sqlite_vec"] == "ok"
    assert report["providers"]["hash"] == "available"
    assert "db_path" in report


def test_indexer_skips_secret_files_and_supports_dry_run(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    safe = memory_dir / "MEMORY.md"
    secret = memory_dir / ".env"
    key = memory_dir / "id_ed25519"
    safe.write_text("This is a safe memory file with enough content to index." * 2)
    secret.write_text("TOKEN=super-secret")
    key.write_text("PRIVATE KEY")

    monkeypatch.setattr(indexer, "MEMORY_DIR", memory_dir)
    planned = indexer.index_memory_files(engine=None, dry_run=True)

    assert str(safe) in planned
    assert all(str(secret) not in item for item in planned)
    assert all(str(key) not in item for item in planned)


def test_batch_insert_records_cached_and_new_embedding_providers_per_chunk(tmp_db):
    engine = MemoryEngine(tmp_db)
    try:
        engine.embedding_mgr.providers = [FixedProvider("hash", [unit_vec(0)])]
        engine.embedding_mgr.get_embeddings(["cached"])

        engine.embedding_mgr.providers = [FixedProvider("gemini", [unit_vec(1)])]
        engine.add_chunk("cached new", store="semantic", source="test", source_type="manual", chunk_size=1)

        rows = engine.conn.execute(
            "SELECT text, embedding_provider FROM chunks WHERE text IN ('cached', 'new') ORDER BY text, created_at"
        ).fetchall()
        providers = {}
        for row in rows:
            providers.setdefault(row["text"], set()).add(row["embedding_provider"])

        assert "hash" in providers["cached"]
        assert "gemini" in providers["new"]
    finally:
        engine.close()
