import type { ReactNode } from 'react';

export type IconName = 'grid' | 'plus' | 'list' | 'briefcase' | 'chevron' | 'arrow' | 'refresh' | 'check' | 'clock' | 'link' | 'file' | 'alert' | 'close' | 'send' | 'logout' | 'activity' | 'copy' | 'search' | 'bookmark' | 'eye' | 'shield' | 'thumbs';

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    list: <><path d="M9 6h12M9 12h12M9 18h12"/><path d="M4 6h.01M4 12h.01M4 18h.01"/></>,
    briefcase: <><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M8 7V5a2 2 0 012-2h4a2 2 0 012 2v2M2 12h20"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    arrow: <><path d="M4 12h16M14 6l6 6-6 6"/></>,
    refresh: <><path d="M20 11a8 8 0 1 0-2 6"/><path d="M20 4v7h-7"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    link: <><path d="M10 13a5 5 0 0 0 7 .5l3-3a5 5 0 0 0-7-7l-2 2"/><path d="M14 11a5 5 0 0 0-7-.5l-3 3a5 5 0 0 0 7 7l2-2"/></>,
    file: <><path d="M6 2h9l5 5v15H6z"/><path d="M15 2v6h5M9 13h8M9 17h8"/></>,
    alert: <><path d="m12 3 10 18H2z"/><path d="M12 9v5M12 18h.01"/></>,
    close: <path d="M18 6 6 18M6 6l12 12"/>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    logout: <><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5M21 12H9"/></>,
    activity: <><path d="M3 12h4l3-7 4 14 3-7h4"/></>,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></>,
    bookmark: <path d="M6 3h12v18l-6-4-6 4z"/>,
    eye: <><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/></>,
    shield: <><path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6z"/><path d="m8 12 3 3 5-6"/></>,
    thumbs: <><path d="M7 10v11H3V10h4Zm0 1 5-8h2v7h6l-1 11H7"/></>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m16 16 5 5"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}
