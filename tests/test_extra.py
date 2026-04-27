import pytest


def test_fastembed_is_lazy(engine):
    fastembed = next(p for p in engine.embedding_mgr.providers if p.name == "fastembed")
    assert fastembed.model is None


def test_recall_special_characters_does_not_crash(engine):
    engine.remember("sqlite fts query edge case", store="semantic")
    results = engine.recall('"unterminated NEAR/5 weird:* query', top_k=3)
    assert isinstance(results, list)


def test_recall_reserved_fts_words_does_not_crash(engine):
    engine.remember("alpha and not near token", store="semantic")
    results = engine.recall("AND NOT NEAR", top_k=3)
    assert isinstance(results, list)


def test_recall_short_terms_does_not_crash(engine):
    engine.remember("ai ux db", store="semantic")
    results = engine.recall("ai ux", top_k=3)
    assert isinstance(results, list)


def test_invalid_store_rejected(engine):
    with pytest.raises(ValueError):
        engine.recall("anything", store="semantic'; DROP TABLE chunks; --")
