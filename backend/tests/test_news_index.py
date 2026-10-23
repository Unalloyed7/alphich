"""Persistent CPU embeddings, immutable version bindings, and read-only API usage."""
import hashlib

import numpy as np
import pytest
from sqlalchemy import func, select

from alphich_kb.embeddings import EncoderUnavailable
from app.news import Event, SourceDocument
from app.intelligence.news_index import CachedNewsEncoder, NewsIndexSource, NewsPassage, index_source, sync_index
from app.intelligence.storage import LLMCall
from app.intelligence.service import normalized_events
from alphich_kb.personalization.selector import NewsSelector
from test_intelligence import store
from test_news_automation import copies


class TinyEncoder:
    dimension = 4
    space_id = "test-space-1"

    def __init__(self):
        self.encoded = []
        self.token_counts = 0

    def count_tokens(self, text):
        self.token_counts += 1
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
    assert encoder.encoded == [] and encoder.token_counts == 0
    sync_index(factory, encoder, ["DEMO_A"], limit=5)
    count = len(encoder.encoded)
    token_counts = encoder.token_counts
    after = client.get("/api/v1/companies/DEMO_A/feed", headers=headers["DEMO_A"]).json()
    assert after["mode"] == "cached_e5_glm" and after["items"] == []
    assert after["extraction_candidates"] == before["extraction_candidates"]
    assert after["news_index"] == before["news_index"]
    assert after["candidate_pool_limit"] == 2000 and len(encoder.encoded) == count
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(NewsPassage)) == 1
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0
    assert encoder.token_counts == token_counts  # HTTP reads do not re-chunk indexed articles.
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


@pytest.mark.parametrize("change", ["revision", "span", "text", "space", "owner", "title", "malformed"])
def test_prepared_spans_cannot_be_reused_for_changed_or_corrupt_sources(store, change):
    _, factory, _, _ = store
    ids = copies(store)
    encoder = TinyEncoder()
    sync_index(factory, encoder, ["DEMO_A"], limit=5)
    with factory.begin() as db:
        source = db.get(SourceDocument, db.get(Event, ids[0]).source_id)
        row = db.scalar(select(NewsIndexSource))
        binding = row.id
        if change == "revision":
            source.version += 1
        elif change == "text":
            source.body = "Иной текст публикации, без изменения остальных служебных полей."
        elif change == "title":
            db.get(Event, ids[0]).title = "Другой заголовок события"
        elif change == "owner":
            row.company_id = "DEMO_B"
        elif change == "malformed":
            row.passages = [None]
        else:
            spans = [dict(span) for span in row.passages]
            if change == "span":
                spans[0]["end"] = 10**9
            else:
                spans[0]["embedding_space_id"] = "different-space"
            row.passages = spans
    with factory() as db:
        event = db.get(Event, ids[0])
        source = db.get(SourceDocument, event.source_id)
        current = normalized_events(db, [(event, source)])
    token_counts, encoded = encoder.token_counts, len(encoder.encoded)
    reader = CachedNewsEncoder(factory, encoder, "DEMO_A", source_bindings={ids[0]: binding})
    with pytest.raises(EncoderUnavailable, match="news_index_"):
        reader.prepared_embeddings(current)
    assert encoder.token_counts == token_counts and len(encoder.encoded) == encoded


def test_prepared_spans_use_current_event_snapshot_after_fact_revision(store):
    _, factory, _, _ = store
    ident = copies(store)[0]
    encoder = TinyEncoder()
    index_source(factory, encoder, ident)
    with factory.begin() as db:
        event = db.get(Event, ident)
        event.version += 1
    with factory() as db:
        event = db.get(Event, ident)
        source = db.get(SourceDocument, event.source_id)
        binding = db.scalar(select(NewsIndexSource.id).where(NewsIndexSource.event_id == ident))
        current = normalized_events(db, [(event, source)])
    token_counts = encoder.token_counts
    selector = NewsSelector(current, CachedNewsEncoder(factory, encoder, "DEMO_A", source_bindings={ident: binding}))
    assert selector.embedding_records
    assert {record["event_version"] for record in selector.embedding_records} == {str(event.version)}
    assert {record["snapshot_hash"] for record in selector.embedding_records} == {selector.events[0]["snapshot_hash"]}
    assert encoder.token_counts == token_counts
