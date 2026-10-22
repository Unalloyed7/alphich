import { useCallback, useEffect, useMemo, useState } from 'react';
import type { DemoSession } from './types';

interface ReadingData { bookmarks: string[]; read: string[]; }
export interface ReadingState {
  isBookmarked: (id: string) => boolean;
  isRead: (id: string) => boolean;
  toggleBookmark: (id: string) => void;
  markRead: (id: string) => void;
}
const empty: ReadingData = { bookmarks: [], read: [] };
function load(key: string): ReadingData {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || 'null');
    const ids = (items: unknown) => Array.isArray(items) ? items.filter((id): id is string => typeof id === 'string' && id.length <= 200).slice(-1000) : [];
    return { bookmarks: ids(value?.bookmarks), read: ids(value?.read) };
  } catch { return empty; }
}
/** Only IDs, scoped by company/role, stored in this tab; never article text or keys. */
export function useReadingState(session: DemoSession | null): ReadingState {
  const key = session ? `alphich.reading.v1:${encodeURIComponent(session.company_id)}:${session.role}` : '';
  const initial = useMemo(() => key ? load(key) : empty, [key]);
  const [snapshot, setSnapshot] = useState({ key, data: initial });
  const data = snapshot.key === key ? snapshot.data : initial;
  useEffect(() => {
    if (!key || snapshot.key !== key) return;
    try { sessionStorage.setItem(key, JSON.stringify(snapshot.data)); } catch { /* Reading remains available in memory. */ }
  }, [key, snapshot]);
  const update = useCallback((id: string, bookmark: boolean) => {
    setSnapshot(previous => {
      const current = previous.key === key ? previous.data : initial;
      const field = bookmark ? 'bookmarks' : 'read';
      if (!bookmark && current.read.includes(id)) return previous.key === key ? previous : { key, data: current };
      const next = current[field].includes(id) ? current[field].filter(item => item !== id) : [...current[field], id].slice(-1000);
      return { key, data: { ...current, [field]: next } };
    });
  }, [key, initial]);
  return {
    isBookmarked: useCallback((id: string) => data.bookmarks.includes(id), [data.bookmarks]),
    isRead: useCallback((id: string) => data.read.includes(id), [data.read]),
    toggleBookmark: useCallback((id: string) => update(id, true), [update]),
    markRead: useCallback((id: string) => update(id, false), [update]),
  };
}
