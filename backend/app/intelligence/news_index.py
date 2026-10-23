"""Persistent, tenant-bound E5 passages. HTTP reads never build missing vectors."""
from functools import lru_cache
import os
import time

from sqlalchemy import JSON, Integer, String, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from app.core import Base
from app.news import Event, SourceDocument
from .provider import hashed

INDEX_VERSION = "news-passages-v1"


class NewsPassage(Base):
    __tablename__ = "news_embedding_passages"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    space_id: Mapped[str] = mapped_column(String(100), index=True)
    input_hash: Mapped[str] = mapped_column(String(64))
    vector: Mapped[list] = mapped_column(JSON)


class NewsIndexSource(Base):
    __tablename__ = "news_index_sources"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(40), index=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    source_id: Mapped[str] = mapped_column(String(40))
    source_version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    space_id: Mapped[str] = mapped_column(String(100))
    passages: Mapped[list] = mapped_column(JSON)
    indexed_at: Mapped[int] = mapped_column(Integer)


@lru_cache(maxsize=2)
def load_encoder(model_dir, model_manifest):
    from alphich_kb.embeddings import E5Encoder, EncoderUnavailable
    try:
        return E5Encoder(model_dir, model_manifest)
    except (EncoderUnavailable, OSError):
        return None


def encoder_for(settings):
    return load_encoder(settings.model_dir, settings.model_manifest)


class CachedNewsEncoder:
    """Cache namespace includes company AND embedding space; no cross-tenant reads."""
    def __init__(self, factory, encoder, company_id, *, writable=False, source_bindings=None):
        self.factory, self.encoder, self.company_id = factory, encoder, company_id
        self.writable = writable
        self.space_id, self.dimension = encoder.space_id, encoder.dimension
        self.hits = self.misses = 0
        self.source_misses = 0
        self.source_bindings = source_bindings

    def count_tokens(self, text):
        return self.encoder.count_tokens(text)

    def encode_query(self, text):
        return self.encoder.encode_query(text)

    def key(self, text):
        return hashed([INDEX_VERSION, self.company_id, self.space_id, hashed(text)])

    def prepared_embeddings(self, events):
        """Read worker-prepared spans instead of tokenizing the corpus on GET.

        Bindings come from the authorized current source rows. Each stored span
        is checked against the current title/body and embedding space; missing
        or stale bindings trigger the existing lexical fallback without writes.
        """
        if self.source_bindings is None or self.writable:
            return None
        from alphich_kb.embeddings import EncoderUnavailable
        keys = [self.source_bindings.get(event["id"]) for event in events]
        indexed = {}
        with self.factory() as db:
            for start in range(0, len(keys), 400):
                for row in db.scalars(select(NewsIndexSource).where(
                        NewsIndexSource.id.in_(keys[start:start + 400]),
                        NewsIndexSource.company_id == self.company_id, NewsIndexSource.space_id == self.space_id)):
                    indexed[row.id] = row
        self.source_misses = sum(key not in indexed for key in keys)
        if self.source_misses:
            raise EncoderUnavailable("news_index_pending")
        records, texts, unique = [], [], {}
        for event, key in zip(events, keys):
            row = indexed[key]
            if event["company_id"] != self.company_id or len(event["sources"]) != 1:
                raise EncoderUnavailable("news_index_source_scope_mismatch")
            source = event["sources"][0]
            if (row.event_id != event["id"] or row.source_id != source["id"]
                    or str(row.source_version) != source["version"]
                    or not isinstance(row.passages, list) or not row.passages):
                raise EncoderUnavailable("news_index_source_revision_mismatch")
            header = f"passage: Новость: {event['title'][:180]}\n"
            for span in row.passages:
                if not isinstance(span, dict):
                    raise EncoderUnavailable("news_index_invalid_span")
                start, end = span.get("start"), span.get("end")
                if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source["text"])
                        or span.get("source_id") != source["id"]
                        or str(span.get("source_version")) != source["version"]
                        or span.get("embedding_space_id") != self.space_id):
                    raise EncoderUnavailable("news_index_invalid_span")
                text = header + source["text"][start:end]
                input_hash = hashed(text)
                if span.get("input_hash") != input_hash or span.get("passage_id") != self.key(text):
                    raise EncoderUnavailable("news_index_source_content_mismatch")
                if input_hash not in unique:
                    unique[input_hash] = len(texts)
                    texts.append(text)
                records.append({"event_id": event["id"], "event_version": event["version"],
                    "snapshot_hash": event["snapshot_hash"], "source_id": source["id"],
                    "source_version": source["version"], "start": start, "end": end,
                    "input_hash": input_hash, "embedding_space_id": self.space_id, "vector_row": unique[input_hash]})
        return records, self.encode_passages(texts)

    def encode_passages(self, texts, batch_size=4):
        import numpy as np
        from alphich_kb.embeddings import EncoderUnavailable
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        keys = [self.key(text) for text in texts]
        found = {}
        with self.factory() as db:
            for start in range(0, len(keys), 400):
                for row in db.scalars(select(NewsPassage).where(NewsPassage.id.in_(keys[start:start+400]),
                        NewsPassage.company_id == self.company_id, NewsPassage.space_id == self.space_id)):
                    found[row.id] = np.asarray(row.vector, dtype=np.float32)
        missing = list(dict.fromkeys(key for key in keys if key not in found))
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        if missing and not self.writable:
            raise EncoderUnavailable("news_index_pending")
        if missing:
            inputs = dict(zip(keys, texts))
            vectors = self.encoder.encode_passages([inputs[key] for key in missing], batch_size=batch_size)
            if vectors.shape != (len(missing), self.dimension):
                raise EncoderUnavailable("news_index_dimension_mismatch")
            with self.factory.begin() as db:
                for key, vector in zip(missing, vectors):
                    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
                        raise EncoderUnavailable("news_index_invalid_vector")
                    try:
                        with db.begin_nested():
                            db.add(NewsPassage(id=key, company_id=self.company_id, space_id=self.space_id,
                                input_hash=hashed(inputs[key]), vector=vector.tolist()))
                            db.flush()
                    except IntegrityError:
                        pass  # Another free indexing worker committed the same immutable input.
                    found[key] = vector
        vectors = np.asarray([found[key] for key in keys], dtype=np.float32)
        if (vectors.shape != (len(keys), self.dimension) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
            raise EncoderUnavailable("news_index_corrupt")
        return vectors


def index_source(factory, encoder, event_id):
    from alphich_kb.personalization.selector import NewsSelector
    from .service import normalized_event
    with factory() as db:
        event = db.get(Event, event_id)
        source = db.get(SourceDocument, event.source_id) if event else None
        if source is None or source.company_id != event.company_id:
            return {"status": "source_unavailable"}
        key = hashed([INDEX_VERSION, event.id, source.id, source.version, source.content_hash,
                      source.title, encoder.space_id])
        if db.get(NewsIndexSource, key) is not None:
            return {"status": "current"}
        normalized = normalized_event(db, event, source)
        identity = (source.id, source.version, source.content_hash, source.title)
        company_id = event.company_id
    cached = CachedNewsEncoder(factory, encoder, company_id, writable=True)
    prepared = NewsSelector([normalized], cached)
    if prepared.encoder_error:
        return {"status": "encoder_unavailable"}
    passages = [{**row, "passage_id": hashed([INDEX_VERSION, company_id, encoder.space_id, row["input_hash"]])}
                for row in prepared.embedding_records]
    with factory.begin() as db:
        current = db.get(SourceDocument, identity[0])
        if current is None or (current.id, current.version, current.content_hash, current.title) != identity:
            return {"status": "source_changed"}
        try:
            with db.begin_nested():
                db.add(NewsIndexSource(id=key, event_id=event_id, company_id=company_id, source_id=identity[0],
                    source_version=identity[1], content_hash=identity[2], space_id=encoder.space_id,
                    passages=passages, indexed_at=int(time.time())))
                db.flush()
        except IntegrityError:
            pass
    return {"status": "indexed", "passages": len(passages), "cache_hits": cached.hits, "new_vectors": cached.misses}


def sync_index(factory, encoder, company_ids, *, limit=5, scan_limit=2000):
    """Index only operator-authorized companies, free CPU work, bounded per tick."""
    if encoder is None:
        return {"status": "encoder_unavailable", "indexed": 0}
    company_ids = sorted(set(company_ids))
    if not company_ids or len(company_ids) > 20 or not 1 <= limit <= 20 or not 1 <= scan_limit <= 5000:
        raise ValueError("invalid_news_index_scope")
    with factory() as db:
        rows = db.execute(select(Event.id, SourceDocument.id, SourceDocument.version,
            SourceDocument.content_hash, SourceDocument.title).join(SourceDocument, Event.source_id == SourceDocument.id).where(
            Event.company_id.in_(company_ids), SourceDocument.company_id == Event.company_id)
            .order_by(SourceDocument.received_at.desc(), Event.id).limit(scan_limit)).all()
        existing = set(db.scalars(select(NewsIndexSource.id).where(NewsIndexSource.company_id.in_(company_ids),
            NewsIndexSource.space_id == encoder.space_id)))
    pending = [row[0] for row in rows if hashed([INDEX_VERSION, *row, encoder.space_id]) not in existing][:limit]
    results = [index_source(factory, encoder, event_id) for event_id in pending]
    return {"status": "ready", "indexed": sum(r["status"] == "indexed" for r in results), "results": results}


def target_ids():
    return sorted({s.strip() for s in os.getenv("DEMO_COMPANY_IDS", "").split(",") if s.strip()})


def main():
    import argparse
    import json
    from app.core import make_session_factory
    from .gateway import load_env_file
    from .settings import Settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--company-id", action="append")
    parser.add_argument("--max-sources", type=int, default=20)
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    companies = args.company_id or target_ids()
    if not companies or not set(companies) <= set(target_ids()) or not 1 <= args.max_sources <= 5000:
        parser.error("Authorized companies and max-sources in 1..5000 required")
    engine, factory = make_session_factory()
    Base.metadata.create_all(engine)
    encoder = encoder_for(Settings.from_env())
    total = 0
    try:
        for _ in range(args.max_sources):
            report = sync_index(factory, encoder, companies, limit=1)
            total += report["indexed"]
            if not report["indexed"]:
                break
        print(json.dumps({"indexed_sources": total, "last_batch": report, "generative_calls": 0}))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
