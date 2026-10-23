"""Bounded SQL work for a growing feed; no network or wall-clock thresholds."""
from sqlalchemy import event as sql_event, select

from app.news import Event, SourceDocument
from app.intelligence.provider import Calls
from app.intelligence.service import normalized_event, normalized_events
from app.intelligence.storage import Extraction
from test_intelligence import store, runtime
from test_news_automation import copies


def test_bulk_facts_keep_revision_checks_without_one_query_per_article(store):
    _, factory, _, settings = store
    first = copies(store)[0]
    worker, _ = runtime(factory, settings)
    worker.extract_event(first, Calls(worker.provider, "bulk-facts-test", lambda: None))
    with factory.begin() as db:
        original = db.get(Event, first)
        source = db.get(SourceDocument, original.source_id)
        for i in range(205):
            source_values = {column.name: getattr(source, column.name) for column in SourceDocument.__table__.columns}
            source_values.update(id=f"source-bulk-{i}", content_hash=f"{i:064x}")
            db.add(SourceDocument(**source_values))
            event_values = {column.name: getattr(original, column.name) for column in Event.__table__.columns}
            event_values.update(id=f"event-bulk-{i}", source_id=source_values["id"])
            db.add(Event(**event_values))
        # A completed result for a different source revision must stay excluded.
        db.add(Extraction(id="stale-extraction", source_id="source-bulk-0", source_version=source.version + 1,
            event_id="event-bulk-0", event_version=original.version, status="completed", facts=[], limitations=[],
            coverage={}, run_id="stale-run", lease_expires_at=0))
    engine = factory.kw["bind"]
    statements = []

    def capture(_, __, statement, *___):
        if "from event_extractions" in statement.lower():
            statements.append(statement)

    with factory() as db:
        rows = db.execute(select(Event, SourceDocument).join(SourceDocument, Event.source_id == SourceDocument.id)
            .where(Event.company_id == "DEMO_A").order_by(Event.id)).all()
        expected = [normalized_event(db, event, source) for event, source in rows]
        sql_event.listen(engine, "before_cursor_execute", capture)
        try:
            actual = normalized_events(db, rows)
        finally:
            sql_event.remove(engine, "before_cursor_execute", capture)
    assert len(actual) == 206
    assert actual == expected
    assert sum(bool(event["facts"]) for event in actual) == 1
    assert len(statements) == 2  # Two bounded batches, rather than 206 lookups.
