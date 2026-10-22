import { Icon, type IconName } from './Icon';
export function AccessibleTabs<T extends string>({ id, label, items, value, onChange, className }: {
  id: string; label: string; items: readonly { id: T; label: string; icon?: IconName }[];
  value: T; onChange: (value: T) => void; className: string;
}) {
  return <div className={className} role="tablist" aria-label={label} onKeyDown={event => {
    const current = items.findIndex(item => item.id === value);
    const next = event.key === 'ArrowRight' ? (current + 1) % items.length : event.key === 'ArrowLeft' ? (current - 1 + items.length) % items.length : event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : -1;
    if (next < 0) return;
    event.preventDefault(); onChange(items[next].id);
    document.getElementById(`${id}-${items[next].id}`)?.focus();
  }}>{items.map(item => <button key={item.id} id={`${id}-${item.id}`} type="button" role="tab" aria-selected={value === item.id} aria-controls={`${id}-panel`} tabIndex={value === item.id ? 0 : -1} className={value === item.id ? 'active' : ''} onClick={() => onChange(item.id)}>{item.icon && <Icon name={item.icon} size={16}/>} {item.label}</button>)}</div>;
}
