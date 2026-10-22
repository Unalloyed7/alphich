import { useCallback, useEffect, useState } from 'react';
import type { FeedFilter } from './personalFeed';
export type Screen = 'dashboard' | 'publish' | 'events' | 'insights' | 'profile';
interface Route { screen: Screen; filter: FeedFilter; eventId: string | null; }
const screens = new Set(['dashboard', 'publish', 'events', 'insights', 'profile']);
const filters = new Set(['all', 'critical', 'bank', 'saved', 'bookmarks', 'unread']);
function read(): Route {
  const [path, query] = window.location.hash.slice(1).split('?');
  const screen = path?.replace(/^\//, '') || 'dashboard';
  const params = new URLSearchParams(query);
  const filter = params.get('filter') || 'all';
  const eventId = params.get('event');
  return { screen: screens.has(screen) ? screen as Screen : 'dashboard', filter: filters.has(filter) ? filter as FeedFilter : 'all', eventId: eventId && eventId.length <= 200 ? eventId : null };
}
function address(route: Route) {
  const params = new URLSearchParams();
  if (route.filter !== 'all') params.set('filter', route.filter);
  if (route.eventId) params.set('event', route.eventId);
  return `#/${route.screen}${params.size ? `?${params}` : ''}`;
}
export function useWorkspaceNavigation() {
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const sync = () => setRoute(read());
    window.addEventListener('popstate', sync);
    window.addEventListener('hashchange', sync);
    return () => { window.removeEventListener('popstate', sync); window.removeEventListener('hashchange', sync); };
  }, []);
  const navigate = useCallback((change: Partial<Route>, replace = false) => {
    const before = read();
    const next = { ...before, ...change };
    const url = address(next);
    if (url !== window.location.hash) {
      const state = { alphichOverlay: !!next.eventId && !before.eventId && !replace };
      window.history[replace ? 'replaceState' : 'pushState'](state, '', url);
    }
    setRoute(next);
  }, []);
  const closeEvent = useCallback(() => {
    if (window.history.state?.alphichOverlay) window.history.back();
    else navigate({ eventId: null }, true);
  }, [navigate]);
  return { route, navigate, closeEvent };
}
