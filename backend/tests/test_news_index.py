"""Persistent CPU embeddings, immutable version bindings, and read-only API usage."""
import hashlib

import numpy as np
import pytest
from sqlalchemy import func, select

from alphich_kb.embeddings import EncoderUnavailable
from app.news import Event, SourceDocument
from app.intelligence.news_index import CachedNewsEncoder, NewsIndexSource, NewsPassage, index_source, sync_index
from app.intelligence.storage import LLMCall
from test_intelligence import store
from test_news_automation import copies


class TinyEncoder:
    dimension = 4
    space_id = "test-space-1"

    def __init__(self):
        self.encoded = []

    def count_tokens(self, text):
        return len(text.split()) + 2

    def encode_passages(self, texts, batch_size=4):
        self.encoded.extend(texts)
        data = np.asarray([[b + 1 for b in hashlib.sha256(text.encode()).digest()[:4]] for text in texts], dtype=np.float32)
        return data / np.linalg.norm(data, axis=1, keepdims=True)

    def encode_query(self, text):
        return np.full(self.dimension, 0.5, dtype=np.float32)


def test_persistent_vectors_survive_encoder_restart_and_do_not_recompute(store):
    _, factory, _, _ = store
    first = TinyEncoder()
    writer = CachedNewsEncoder(factory, first, "DEMO_A", writable=True)
    wanted = writer.encode_passages(["passage: На рынке изменились условия"])
    assert len(first.encoded) == 1
    restarted = TinyEncoder()
    reader = CachedNewsEncoder(factory, restarted, "DEMO_A")
    assert np.allclose(reader.encode_passages(["passage: На рынке изменились условия"]), wanted)
    assert not restarted.encoded and reader.hits == 1


def test_identical_private_text_does_not_share_tenant_cache_or_embedding_space(store):
    _, factory, _, _ = store
    encoder = TinyEncoder()
    text = ["passage: Приватная публикация"]
    CachedNewsEncoder(factory, encoder, "DEMO_A", writable=True).encode_passages(text)
    with pytest.raises(EncoderUnavailable, match="pending"):
        CachedNewsEncoder(factory, encoder, "DEMO_B").encode_passages(text)
    other = TinyEncoder(); other.space_id = "test-space-2"
    with pytest.raises(EncoderUnavailable, match="pending"):
        CachedNewsEncoder(factory, other, "DEMO_A").encode_passages(text)


def test_scope_checked_before_indexing_and_source_spans_are_persisted(store):
    _, factory, _, _ = store
    ids = copies(store)
    encoder = TinyEncoder()
    report = sync_index(factory, encoder, ["DEMO_A"], limit=5)
    assert report["indexed"] == 1
    with factory() as db:
        row = db.scalar(select(NewsIndexSource))
        assert row.company_id == "DEMO_A" and row.event_id == ids[0]
        assert row.passages[0]["start"] == 0
        assert db.get(NewsPassage, row.passages[0]["passage_id"]).company_id == "DEMO_A"
    before = len(encoder.encoded)
    assert sync_index(factory, encoder, ["DEMO_A"], limit=5)["indexed"] == 0
    assert len(encoder.encoded) == before


def test_new_source_revision_creates_new_binding_and_vector_without_resetting_old_data(store):
    _, factory, _, _ = store
    ids = copies(store)
    encoder = TinyEncoder()
    index_source(factory, encoder, ids[0])
    with factory.begin() as db:
        source = db.get(SourceDocument, db.get(Event, ids[0]).source_id)
        source.version += 1
        source.body += " Подтверждены дополнительные условия."
        source.content_hash = "b" * 64
    assert index_source(factory, encoder, ids[0])["status"] == "indexed"
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(NewsIndexSource)) == 2
        assert db.scalar(select(func.count()).select_from(NewsPassage)) == 2


def test_api_reads_do_not_infer_or_grade_from_a_completed_index(store, monkeypatch):
    client, factory, headers, _ = store
    ids = copies(store)
    encoder = TinyEncoder()
    def forbidden_encoder(*args, **kwargs):
        raise AssertionError("Reading the feed must not load an encoder")
    monkeypatch.setattr("app.intelligence.news_index.encoder_for", forbidden_encoder)
    before = client.get("/api/v1/companies/DEMO_A/feed", headers=headers["DEMO_A"]).json()
    assert before["news_index"] == {"builds_on_read": False, "inference_on_read": False}
    assert before["mode"] == "cached_e5_glm" and before["items"] == []
    assert before["trace"]["model_calls_on_read"] == 0
    assert before["extraction_candidates"][0]["event_id"] == ids[0]
    assert before["extraction_candidates"][0]["relevance_grade"] is None
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(NewsPassage)) == 0
    assert encoder.encoded == []
    sync_index(factory, encoder, ["DEMO_A"], limit=5)
    count = len(encoder.encoded)
    after = client.get("/api/v1/companies/DEMO_A/feed", headers=headers["DEMO_A"]).json()
    assert after["mode"] == "cached_e5_glm" and after["items"] == []
    assert after["extraction_candidates"] == before["extraction_candidates"]
    assert after["news_index"] == before["news_index"]
    assert after["candidate_pool_limit"] == 2000 and len(encoder.encoded) == count
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(NewsPassage)) == 1
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0
    assert client.get("/api/v1/companies/DEMO_A/feed", headers=headers["DEMO_B"]).status_code == 403


def test_corrupt_cached_vector_cannot_trigger_inference_or_a_fallback_grade_on_get(store, monkeypatch):
    client, factory, headers, _ = store
    copies(store)
    encoder = TinyEncoder()
    sync_index(factory, encoder, ["DEMO_A"], limit=5)
    count = len(encoder.encoded)
    with factory.begin() as db:
        db.scalar(select(NewsPassage)).vector = [0, 0, 0, 0]
    def forbidden_encoder(*args, **kwargs):
        raise AssertionError("Reading the feed must not load or repair embeddings")
    monkeypatch.setattr("app.intelligence.news_index.encoder_for", forbidden_encoder)
    feed = client.get("/api/v1/companies/DEMO_A/feed", headers=headers["DEMO_A"]).json()
    assert feed["mode"] == "cached_e5_glm" and feed["items"] == []
    assert feed["news_index"] == {"builds_on_read": False, "inference_on_read": False}
    assert feed["trace"]["model_calls_on_read"] == 0
    assert feed["extraction_candidates"][0]["relevance_grade"] is None
    with factory() as db:
        assert db.scalar(select(NewsPassage)).vector == [0, 0, 0, 0]
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0
    assert len(encoder.encoded) == count
