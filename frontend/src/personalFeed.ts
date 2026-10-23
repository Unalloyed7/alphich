import { useCallback, useEffect, useRef, useState } from 'react';
import { analysisApi } from './api';
import type { FeedItem, FeedResult } from './analysis';
import type { DemoSession } from './types';

const REFRESH_MS = 20_000;
const MAX_RETRY_DELAY_MS = 120_000;

export type FeedFilter = 'all' | 'critical' | 'bank' | 'saved' | 'bookmarks' | 'unread';
export interface PersonalFeedState {
  feed: FeedResult | null;
  loading: boolean;
  error: string | null;
  updatedAt: number | null;
  reload: (silent?: boolean) => Promise<void>;
}

/** One read-only source for dashboard widgets, feed cards and event badges. */
export function usePersonalFeed(session: DemoSession | null): PersonalFeedState {
  const token = session?.token;
  const companyId = session?.company_id;
  const activeToken = useRef(token);
  activeToken.current = token;
  const request = useRef(0);
  const inFlight = useRef<{ token: string; promise: Promise<FeedResult> } | null>(null);
  const retry = useRef<{ token?: string; failures: number; after: number }>({ failures: 0, after: 0 });
  const [snapshot, setSnapshot] = useState<{
    token?: string; feed: FeedResult | null; error: string | null; updatedAt: number | null;
  }>({ feed: null, error: null, updatedAt: null });
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async (silent = false) => {
    if (!token || !companyId) return;
    if (silent && retry.current.token === token && Date.now() < retry.current.after) return;
    const current = ++request.current;
    if (!silent) setLoading(true);
    // Timer ticks and manual refreshes share a pending request. A slow server
    // must not receive another feed calculation every twenty seconds.
    let pending = inFlight.current;
    if (!pending || pending.token !== token) {
      pending = { token, promise: analysisApi.feed(companyId, token) };
      inFlight.current = pending;
    }
    try {
      const feed = await pending.promise;
      if (current === request.current && activeToken.current === token) {
        retry.current = { token, failures: 0, after: 0 };
        setSnapshot({ token, feed, error: null, updatedAt: Date.now() });
      }
    } catch (error) {
      if (current === request.current && activeToken.current === token) {
        const failures = retry.current.token === token ? retry.current.failures + 1 : 1;
        retry.current = {
          token, failures,
          after: Date.now() + Math.min(REFRESH_MS * 2 ** Math.min(failures, 3), MAX_RETRY_DELAY_MS),
        };
        setSnapshot(previous => ({
          token, feed: previous.token === token ? previous.feed : null,
          updatedAt: previous.token === token ? previous.updatedAt : null,
          error: error instanceof Error ? error.message : 'Не удалось обновить аналитику',
        }));
      }
    } finally {
      if (inFlight.current === pending) inFlight.current = null;
      if (current === request.current && activeToken.current === token) setLoading(false);
    }
  }, [companyId, token]);

  useEffect(() => {
    if (!token) { setLoading(false); return; }
    void reload();
    const refresh = () => { if (document.visibilityState !== 'hidden') void reload(true); };
    const timer = window.setInterval(refresh, REFRESH_MS);
    document.addEventListener('visibilitychange', refresh);
    return () => {
      request.current++;
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', refresh);
    };
  }, [token, reload]);

  const current = snapshot.token === token;
  return {
    feed: current ? snapshot.feed : null,
    error: current ? snapshot.error : null,
    updatedAt: current ? snapshot.updatedAt : null,
    loading: (!current && !!token) || loading,
    reload,
  };
}

/** Counts refer only to this returned feed, never to all news or company risk. */
export function summarizeFeed(feed: FeedResult | null) {
  const items = feed?.items ?? [];
  const ranked = [...items].sort((a, b) => (b.relevance_grade ?? -1) - (a.relevance_grade ?? -1));
  const actions = ranked.filter(item => item.relevance_assessment?.required_action?.trim());
  const uniqueActions: FeedItem[] = [];
  const seen = new Set<string>();
  for (const item of actions) {
    const text = item.relevance_assessment!.required_action!.trim();
    if (!seen.has(text)) { seen.add(text); uniqueActions.push(item); }
  }
  return {
    total: items.length,
    critical: items.filter(item => item.relevance_grade === 5).length,
    high: items.filter(item => item.relevance_grade === 4).length,
    other: items.filter(item => item.relevance_grade !== 4 && item.relevance_grade !== 5).length,
    saved: items.filter(item => item.insight).length,
    bank: items.filter(item => item.insight?.bank_view?.status === 'found').length,
    historical: items.filter(item => item.insight?.bank_view?.status === 'historical_only').length,
    lead: ranked[0],
    actions: uniqueActions,
  };
}
