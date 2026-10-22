import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon, type IconName } from './Icon';
import type { FeedItem } from './analysis';
import type { Event } from './types';
import type { Screen } from './navigation';

const sections: { screen: Screen; title: string; description: string; icon: IconName }[] = [
  { screen: 'dashboard', title: 'Обзор', description: 'Сводка и важное для бизнеса', icon: 'grid' },
  { screen: 'insights', title: 'Моя аналитика', description: 'Персональная лента и фильтры', icon: 'activity' },
  { screen: 'events', title: 'Все события', description: 'Сохранённые публикации и источники', icon: 'list' },
  { screen: 'profile', title: 'Моя компания', description: 'Профиль и основания персонализации', icon: 'briefcase' },
  { screen: 'publish', title: 'Добавить новость', description: 'Новая публикация для разбора', icon: 'plus' },
];
export function QuickSearch({ items, events, onClose, onScreen, onEvent }: {
  items: FeedItem[]; events: Event[]; onClose: () => void;
  onScreen: (screen: Screen) => void; onEvent: (id: string) => void;
}) {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const close = useRef(onClose); close.current = onClose;
  const results = useMemo(() => {
    const text = query.trim().toLocaleLowerCase('ru-RU');
    const links = sections.map(item => ({ key: `section-${item.screen}`, title: item.title, description: item.description, icon: item.icon, screen: item.screen, eventId: null as string | null }));
    const seen = new Set<string>();
    const news = [...items.map(item => ({ id: item.event_id, title: item.insight?.headline || item.title || 'Событие', description: item.relevance_assessment?.business_channel || 'Событие из персональной ленты' })), ...events.map(item => ({ id: item.id, title: item.title, description: 'Сохранённая публикация' }))].filter(item => { if (seen.has(item.id)) return false; seen.add(item.id); return true; }).map(item => ({ key: `event-${item.id}`, title: item.title, description: item.description, icon: 'file' as IconName, screen: null as Screen | null, eventId: item.id }));
    return [...links, ...news].filter(item => !text || `${item.title} ${item.description}`.toLocaleLowerCase('ru-RU').includes(text)).slice(0, 10);
  }, [query, items, events]);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden'; input.current?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close.current(); }
      if (event.key === 'Tab') {
        const focusable = [...(dialog.current?.querySelectorAll<HTMLElement>('button:not([tabindex="-1"]), input') ?? [])];
        const first = focusable[0], last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => { document.body.style.overflow = overflow; window.removeEventListener('keydown', handleKey); if (previous?.isConnected) previous.focus(); };
  }, []);
  useEffect(() => { document.getElementById(`search-result-${active}`)?.scrollIntoView({ block: 'nearest' }); }, [active]);
  const choose = (index: number) => {
    const item = results[index]; if (!item) return;
    onClose(); if (item.eventId) onEvent(item.eventId); else if (item.screen) onScreen(item.screen);
  };
  return <div className="search-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}><div ref={dialog} className="quick-search" role="dialog" aria-modal="true" aria-labelledby="quick-search-title"><div className="quick-search-heading"><h2 id="quick-search-title">Быстрый поиск</h2><button type="button" className="icon-button" onClick={onClose} aria-label="Закрыть поиск"><Icon name="close" size={20}/></button></div><div className="quick-search-input"><Icon name="search" size={21}/><label className="sr-only" htmlFor="quick-search-input">Событие или раздел</label><input ref={input} id="quick-search-input" value={query} role="combobox" aria-expanded="true" aria-controls="search-results" aria-autocomplete="list" aria-activedescendant={results[active] ? `search-result-${active}` : undefined} placeholder="Найти событие или перейти в раздел…" onChange={event => { setQuery(event.target.value); setActive(0); }} onKeyDown={event => {
    if (event.nativeEvent.isComposing) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setActive(index => results.length ? (index + (event.key === 'ArrowDown' ? 1 : -1) + results.length) % results.length : 0); }
    if (event.key === 'Enter') { event.preventDefault(); choose(active); }
  }}/></div><p className="search-scope">По загруженным событиям и разделам вашего пространства.</p><div className="quick-search-results" id="search-results" role="listbox" aria-label="Результаты быстрого поиска">{results.map((item, index) => <button id={`search-result-${index}`} key={item.key} role="option" aria-selected={active === index} tabIndex={-1} type="button" className={active === index ? 'active' : ''} onMouseMove={() => setActive(index)} onClick={() => choose(index)}><span className="search-result-icon"><Icon name={item.icon} size={18}/></span><span><strong>{item.title}</strong><small>{item.description}</small></span><Icon name="arrow" size={16}/></button>)}</div>{results.length === 0 && <div className="search-empty" role="status"><Icon name="search" size={25}/><strong>Ничего не найдено</strong><p>Попробуйте другое название. Поиск ограничен уже загруженными событиями.</p></div>}<div className="search-keyboard-hints"><span><kbd>↑</kbd><kbd>↓</kbd> выбрать</span><span><kbd>Enter</kbd> открыть</span><span><kbd>Esc</kbd> закрыть</span></div></div></div>;
}
