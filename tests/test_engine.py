import os
import sqlite3

from ham.memory_engine import MemoryEngine, HashProvider



def test_init_creates_tables(engine, tmp_db):
    conn = sqlite3.connect(tmp_db)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {t[0] for t in tables}
    assert "chunks" in names
    assert "embedding_cache" in names
    assert "semantic_links" in names
    assert "consolidation_log" in names
    conn.close()


def test_remember_and_recall(engine):
    cid = engine.remember(
        text="User prefers dark mode",
        store="semantic",
        source="test",
        importance=0.8,
    )
    assert cid is not None
    results = engine.recall("dark mode", top_k=5)
    texts = [r["text"] for r in results]
    assert any("dark mode" in t for t in texts)


def test_stats(engine):
    engine.remember("test fact", store="semantic")
    stats = engine.get_stats()
    assert stats["total_chunks"] >= 1
    assert stats["by_store"]["semantic"] >= 1


def test_hash_fallback_returns_vectors():
    hf = HashProvider(dims=3072)
    vecs = hf.embed(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 3072
    assert len(vecs[1]) == 3072
    # deterministic
    v1a = hf.embed(["hello"])[0]
    v1b = hf.embed(["hello"])[0]
    assert v1a == v1b


def test_consolidate_episodic_no_crash(engine):
    engine.remember("old session text", store="episodic")
    engine.consolidate_episodic(days=0)
    stats = engine.get_stats()
    # should not raise; archive may or may not have entries depending on logic
    assert isinstance(stats, dict)


def test_embedding_manager_no_key_uses_hash_fallback(tmp_db):
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)
    engine = MemoryEngine(tmp_db)
    mgr = engine.embedding_mgr
    # force only hash provider
    mgr.providers = [mgr.providers[-1]]  # HashProvider is last
    vecs, provider = mgr.get_embeddings(["foo"])
    assert provider == "hash"
    assert len(vecs[0]) == 3072
    engine.close()


def test_deduplicate_semantic_removes_orphan_vectors(engine):
    engine.remember("duplicate semantic memory", store="semantic")
    engine.remember("duplicate semantic memory memory", store="semantic")

    result = engine.deduplicate_semantic(similarity_threshold=0.9)

    orphan_count = engine.conn.execute(
        "SELECT COUNT(*) FROM chunks_vec WHERE rowid NOT IN (SELECT rowid FROM chunks)"
    ).fetchone()[0]
    assert result["count"] >= 1
    assert orphan_count == 0
