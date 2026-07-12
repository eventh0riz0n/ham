"""Tests for HAM v2 store, extraction, and provider lifecycle.

Run from the hermes-agent repo root (needs agent.memory_provider importable):

    cd ~/.hermes/hermes-agent && venv/bin/python -m pytest \
        ~/.hermes/plugins/ham/tests/test_ham_v2.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent


def _load_plugin():
    """Load the plugin package the same way plugins/memory/__init__.py does."""
    name = "_hermes_user_memory.ham"
    if name in sys.modules:
        return sys.modules[name]
    import importlib.machinery
    parent_spec = importlib.machinery.ModuleSpec("_hermes_user_memory", None, is_package=True)
    parent_spec.submodule_search_locations = []
    parent = importlib.util.module_from_spec(parent_spec)
    sys.modules.setdefault("_hermes_user_memory", parent)
    spec = importlib.util.spec_from_file_location(
        name, str(PLUGIN_DIR / "__init__.py"),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ham = _load_plugin()
store_mod = sys.modules["_hermes_user_memory.ham.store"]
extract_mod = sys.modules["_hermes_user_memory.ham.extract"]


class FakeEmbedder:
    """Deterministic embedder: bag-of-words md5-bucket projection.

    md5, not the built-in hash() — string hashing is salted per process
    (PYTHONHASHSEED), which made similarity scores flap across runs and
    fail the prefetch threshold in CI. No model download.
    """

    model_name = "fake"

    def warm(self):
        return True

    def embed(self, texts):
        import hashlib
        import numpy as np
        out = []
        for t in texts:
            vec = np.zeros(64, dtype="float32")
            for w in t.lower().split():
                bucket = int(hashlib.md5(w.encode()).hexdigest(), 16) % 64
                vec[bucket] += 1.0
            n = np.linalg.norm(vec)
            out.append(list((vec / n) if n else vec))
        return out


class DeadEmbedder:
    model_name = "dead"

    def warm(self):
        return False

    def embed(self, texts):
        return None


@pytest.fixture
def store(tmp_path):
    s = store_mod.HamStore(tmp_path / "test.db", embed_model="fake")
    s.embedder = FakeEmbedder()
    yield s
    s.close()


# -- store ---------------------------------------------------------------------

def test_add_and_recall(store):
    store.add_fact("Beniamin preferuje język polski w odpowiedziach", kind="user_pref")
    store.add_fact("Pulse działa na porcie 8789", kind="infra")
    results = store.search("jaki port ma Pulse")
    assert results
    assert "8789" in results[0]["text"]


def test_exact_duplicate_is_touched_not_readded(store):
    a = store.add_fact("fakt testowy o dedupie", kind="note")
    b = store.add_fact("fakt testowy o dedupie", kind="note")
    assert a == b
    assert store.stats()["active"] == 1


def test_supersede_keeps_history(store):
    old = store.add_fact("Primary model to GLM 4.9", kind="infra")
    new = store.supersede(old, "Primary model to GLM 5.2", session_id="s1")
    assert new and new != old
    old_row = store.get(old)
    assert old_row["status"] == "superseded"
    assert old_row["superseded_by"] == new
    active = store.search("primary model GLM")
    assert all(r["id"] != old for r in active)
    with_history = store.search("primary model GLM", include_superseded=True)
    assert any(r["id"] == old for r in with_history)


def test_invalidate(store):
    fid = store.add_fact("Dentysta 30.10.2026", kind="note")
    assert store.invalidate(fid, reason="appointment done")
    assert store.get(fid)["status"] == "superseded"


def test_bm25_only_when_embedder_dead(store):
    store.add_fact("UpHOTEL tender MVP w katalogu uphotel-tenders", kind="project")
    store.embedder = DeadEmbedder()
    results = store.search("uphotel tenders")
    assert results
    assert results[0]["score_parts"]["vector"] is None


def test_polish_diacritics_fts(store):
    store.add_fact("Wentylator MacBooka sterowany przez macsmc", kind="infra")
    # remove_diacritics 2 → query without Polish diacritics still matches
    results = store.search("wentylator macbooka")
    assert results


def test_episode_excluded_from_default_search(store):
    store.add_fact("Sesja o konfiguracji cronów", kind="episode", session_id="sess_x")
    store.add_fact("Crony hermesa zarządzane przez hermes cron", kind="infra")
    default = store.search("crony konfiguracja")
    assert all(r["kind"] != "episode" for r in default)
    with_eps = store.search("crony konfiguracja", include_episodes=True)
    assert any(r["kind"] == "episode" for r in with_eps)


def test_consolidate_dedups_near_duplicates(store):
    store.add_fact("Beniamin lubi krótkie odpowiedzi po polsku", kind="user_pref")
    store.add_fact("Beniamin lubi krótkie odpowiedzi po polsku zawsze", kind="user_pref")
    report = store.consolidate(sim_threshold=0.8)
    assert report["merged"] == 1
    assert store.stats()["active"] == 1


def test_consolidate_prunes_old_superseded(store):
    old = store.add_fact("stary fakt", kind="note")
    store.invalidate(old)
    store.conn.execute(
        "UPDATE facts SET invalidated_at = ? WHERE id = ?",
        (int(time.time()) - 200 * 86400, old),
    )
    store.conn.commit()
    report = store.consolidate()
    assert report["pruned"] == 1
    assert store.get(old) is None


# -- extraction ------------------------------------------------------------------

def _turns():
    return [
        ("Zapamiętaj że przechodzimy z portu 8789 na 8800 dla Pulse",
         "Jasne, zapamiętane — Pulse będzie na porcie 8800."),
        ("I wolę żeby raporty były w formie tabeli",
         "OK, będę używał tabel w raportach."),
    ]


def test_extraction_applies_operations(store):
    old = store.add_fact("Pulse działa na porcie 8789", kind="infra")

    def fake_llm(instructions, payload):
        assert "EXISTING FACTS" in payload and "8789" in payload
        return json.dumps({
            "operations": [
                {"op": "supersede", "id": old, "text": "Pulse działa na porcie 8800", "kind": "infra"},
                {"op": "add", "text": "Beniamin woli raporty w formie tabeli",
                 "kind": "user_pref", "importance": 0.7},
                {"op": "noop"},
            ],
            "session_summary": "Zmiana portu Pulse na 8800 i preferencja tabel w raportach.",
        })

    report = extract_mod.extract_and_reconcile(
        store, _turns(), session_id="sess_1", llm_caller=fake_llm)
    assert report["ran"] and report["ops_applied"] == 2
    assert len(report["superseded"]) == 1 and len(report["added"]) == 1
    assert report["episode"] is not None
    top = store.search("port Pulse")
    assert "8800" in top[0]["text"]
    eps = store.list_facts(kind="episode")
    assert eps and eps[0]["session_id"] == "sess_1"


def test_extraction_ignores_bogus_ops(store):
    def bad_llm(instructions, payload):
        return json.dumps({"operations": [
            {"op": "supersede", "id": 99999, "text": "hack"},   # unknown id
            {"op": "update", "id": None, "text": "x"},
            {"op": "add", "text": ""},                            # empty
            {"op": "delete", "text": "not an op"},
        ]})
    report = extract_mod.extract_and_reconcile(
        store, _turns(), session_id="s", llm_caller=bad_llm)
    assert report["ran"] and report["ops_applied"] == 0


def test_extraction_survives_unparseable_output(store):
    report = extract_mod.extract_and_reconcile(
        store, _turns(), session_id="s", llm_caller=lambda i, p: "sorry, no JSON here")
    assert not report["ran"] and "unparseable" in report["skipped_reason"]


def test_extraction_skips_short_sessions(store):
    report = extract_mod.extract_and_reconcile(
        store, [("hej", "cześć")], session_id="s", llm_caller=None)
    assert not report["ran"]


def test_digest_truncation():
    turns = [("x" * 3000, "y" * 3000)] * 10
    digest = extract_mod.build_digest(turns, max_chars=5000)
    assert len(digest) <= 5100
    assert "skrócono" in digest


def test_turns_from_messages():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "pytanie"},
        {"role": "assistant", "content": [{"type": "text", "text": "odpowiedź"}]},
        {"role": "tool", "content": "tool output"},
        {"role": "user", "content": "drugie pytanie"},
    ]
    turns = extract_mod.turns_from_messages(messages)
    assert turns[0] == ("pytanie", "odpowiedź")
    assert turns[1][0] == "drugie pytanie"


def test_search_match_score_ignores_recency_importance(store):
    """A fresh, max-importance fact must not pass the injection gate on
    freshness alone — match_score reflects query-relatedness only."""
    fid = store.add_fact("Preferencje treningowe: plan MOVE i okno jedzenia",
                         kind="user_pref", importance=1.0)
    results = store.search("konfiguracja portów serwera produkcyjnego", top_k=10)
    for r in results:
        if r["id"] == fid:
            assert r["match_score"] < 0.5
            assert r["score"] > r["match_score"]  # rec+imp inflate legacy score


def test_search_results_ordered_by_rrf(store):
    store.add_fact("Mail Triage działa pod adresem 8791", kind="infra")
    store.add_fact("Zupełnie inny fakt o kotach domowych", kind="note")
    results = store.search("mail triage adres", top_k=5)
    rrfs = [r["rrf"] for r in results]
    assert rrfs == sorted(rrfs, reverse=True)
    assert results[0]["rrf"] > 0


# -- prefetch query construction ---------------------------------------------------

def test_build_query_long_message_passes_through():
    msg = "x" * 120
    assert ham.build_prefetch_query(msg, "poprzednia", "odpowiedź") == msg


def test_build_query_short_message_gets_context():
    q = ham.build_prefetch_query("popraw to", "konfiguracja watchera IMAP", "Watcher IMAP ma timeout")
    assert "popraw to" in q and "IMAP" in q


def test_build_query_trivial_without_context_is_empty():
    assert ham.build_prefetch_query("ok", "", "") == ""
    assert ham.build_prefetch_query("", "", "") == ""


# -- provider lifecycle ------------------------------------------------------------

@pytest.fixture
def provider(tmp_path):
    p = ham.HamMemoryProvider(config={
        "db_path": str(tmp_path / "prov.db"),
        # Nonexistent model name: the background warm thread fails fast
        # instead of downloading the real model (matters in CI); every test
        # swaps in FakeEmbedder anyway.
        "embed_model": "test/nonexistent-model",
        "extract_enabled": True,
        # The 0.50 default is calibrated to the real ~150-fact store; tiny
        # test corpora produce near-zero BM25 magnitudes (idf ~ 0), so pin a
        # mechanics-level gate — these tests exercise behavior, not calibration.
        "prefetch_min_match": 0.25,
    })
    p.initialize("sess_A", hermes_home=str(tmp_path), platform="cli",
                 agent_context="primary")
    p._store.embedder = FakeEmbedder()
    yield p
    p.shutdown()


def test_provider_prefetch_roundtrip(provider):
    provider._store.add_fact("Mail Triage działa pod 100.109.206.101:8791", kind="infra")
    block = provider.prefetch("gdzie jest mail triage")
    assert "8791" in block and "HAM recall" in block


def test_provider_prefetch_empty_on_no_match(provider):
    assert provider.prefetch("zupełnie niezwiązane zapytanie o kosmitach xyzzy") == ""


def test_provider_prefetch_dedups_recent_injections(provider):
    provider._store.add_fact("Mail Triage działa pod 100.109.206.101:8791", kind="infra")
    first = provider.prefetch("gdzie jest mail triage")
    assert "8791" in first
    second = provider.prefetch("gdzie jest mail triage")
    assert "8791" not in second  # already in context from previous turn


def test_provider_short_message_enriched_from_previous_turn(provider):
    provider._store.add_fact(
        "Watcher IMAP przetargów ma timeout na socket", kind="infra")
    # Filler corpus: single-document FTS5 has idf=0 (BM25 lane scores 0);
    # a few unrelated rows restore realistic term weighting.
    provider._store.add_fact("Dentysta wizyta w październiku", kind="note")
    provider._store.add_fact("Raporty tygodniowe idą na Telegram", kind="decision")
    provider._store.add_fact("Backup rclone iCloud działa poprawnie", kind="infra")
    provider.sync_turn(
        "co się dzieje z watcherem IMAP od przetargów?",
        "Watcher IMAP działa, sprawdzam szczegóły timeoutów na sockecie.",
        session_id="sess_A")
    block = provider.prefetch("a napraw to")
    assert "Watcher IMAP" in block


def test_provider_tool_recall_and_remember(provider):
    res = json.loads(provider.handle_tool_call("ham_memory", {
        "action": "remember", "text": "Testowy fakt narzędziowy", "kind": "note"}))
    assert res["stored"]
    res = json.loads(provider.handle_tool_call("ham_memory", {
        "action": "recall", "query": "fakt narzędziowy"}))
    assert res["count"] >= 1
    res = json.loads(provider.handle_tool_call("ham_memory", {
        "action": "forget", "fact_id": res["results"][0]["id"]}))
    assert res["forgotten"]


def test_provider_buffers_and_extracts_on_session_end(provider, monkeypatch):
    calls = {}

    def fake_caller(prov, model):
        def _c(instructions, payload):
            calls["payload"] = payload
            return json.dumps({"operations": [
                {"op": "add", "text": "Nowy fakt z sesji", "kind": "note"}],
                "session_summary": None})
        return _c

    monkeypatch.setattr(extract_mod, "default_llm_caller", fake_caller)
    provider.sync_turn(
        "pierwsza wiadomość użytkownika w której ustalamy że nowy projekt "
        "będzie hostowany na maszynie ben-mba i używał portu 9100",
        "Przyjąłem — projekt będzie na ben-mba, port 9100, dodaję do konfiguracji "
        "systemd i aktualizuję dokumentację procesu zgodnie z konwencjami.",
        session_id="sess_A")
    provider.sync_turn(
        "druga wiadomość która coś ustala: raporty tygodniowe mają trafiać na "
        "Telegram zamiast do plików lokalnych, zapamiętaj to na przyszłość",
        "Jasne, zapamiętane — raporty tygodniowe będą wysyłane na Telegram.",
        session_id="sess_A")
    provider.on_session_end([])
    assert "pierwsza wiadomość" in calls["payload"]
    assert provider._store.search("nowy fakt z sesji", touch=False)


def test_provider_skips_extraction_for_cron_context(tmp_path, monkeypatch):
    p = ham.HamMemoryProvider(config={"db_path": str(tmp_path / "cron.db"),
                                      "embed_model": "test/nonexistent-model"})
    p.initialize("sess_C", hermes_home=str(tmp_path), agent_context="cron")
    p._store.embedder = FakeEmbedder()
    p.sync_turn("cron user msg", "cron reply", session_id="sess_C")
    p.sync_turn("more", "more", session_id="sess_C")
    called = []
    monkeypatch.setattr(extract_mod, "extract_and_reconcile",
                        lambda *a, **k: called.append(1))
    p.on_session_end([])
    assert not called
    p.shutdown()


def test_provider_no_double_extraction(provider, monkeypatch):
    count = []
    monkeypatch.setattr(extract_mod, "extract_and_reconcile",
                        lambda *a, **k: (count.append(1), {"ran": True})[1])
    provider.sync_turn("a", "b", session_id="sess_A")
    provider.sync_turn("c", "d", session_id="sess_A")
    provider.on_session_end([])
    provider.on_session_end([])  # shutdown path fires it again
    assert len(count) == 1


def test_provider_mirrors_builtin_memory_writes(provider):
    provider.on_memory_write("add", "user", "Beniamin pracuje w strefie Europe/Warsaw")
    facts = provider._store.search("strefa czasowa Beniamin", touch=False)
    assert facts and facts[0]["kind"] == "user_pref"
    assert facts[0]["source"] == "builtin"


def test_provider_session_switch_carries_buffer(provider):
    provider.sync_turn("msg", "reply", session_id="sess_A")
    provider.on_session_switch("sess_B", parent_session_id="sess_A", reset=False)
    assert provider._turn_buffer.get("sess_B")
    assert provider._session_id == "sess_B"
