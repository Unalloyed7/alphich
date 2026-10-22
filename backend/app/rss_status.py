"""Authenticated, read-only schedule and heartbeat for the continuous collector."""
import os
import time

from fastapi import Depends, HTTPException

from app.rss import (CollectorHeartbeat, FeedState, DEFAULT_CONFIG, collector_enabled, load_config)
from app.rss_fetch import CollectionError


def register_status_route(app, get_db, current_session):
    @app.get('/api/v1/news-collection/status')
    def collection_status(_=Depends(current_session), db=Depends(get_db)):
        try:
            feeds = load_config(os.getenv('RSS_CONFIG', str(DEFAULT_CONFIG)))
            enabled = collector_enabled()
        except CollectionError as error:
            raise HTTPException(503, error.code) from None
        heartbeat = db.get(CollectorHeartbeat, 'main')
        now = int(time.time())
        state = ('disabled' if not enabled else 'unknown' if heartbeat is None else
                 'running' if heartbeat.enabled and now - heartbeat.seen_at <= 60 else 'stale')
        override = os.getenv('RSS_POLL_MINUTES', '').strip()
        return {'enabled': enabled, 'state': state,
            'heartbeat_at': heartbeat.seen_at if heartbeat else None,
            'poll_minutes': int(override) if override else None,
            'feeds': [{'id': feed.id, 'url': feed.url, 'poll_seconds': feed.poll_seconds,
                       'next_poll_at': min(record.next_poll_at, record.last_checked_at + feed.poll_seconds)
                           if record and record.last_checked_at is not None else record.next_poll_at if record else None,
                       'last_checked_at': record.last_checked_at if record else None,
                       'last_success_at': record.last_success_at if record else None,
                       'failures': record.failures if record else 0,
                       'last_error': record.last_error if record else None}
                      for feed in feeds for record in [db.get(FeedState, feed.id)]],
            'note': 'Новые публикации получаем после появления в лентах источников и очередного опроса. '
                    'Состояние процесса определяется heartbeat, доступность каждого сайта показана отдельно.'}
