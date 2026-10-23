import { Icon } from './Icon';
import { roles } from './roles';
import { summarizeFeed, type FeedFilter, type PersonalFeedState } from './personalFeed';
import type { DemoSession } from './types';
import './dashboard.css';

interface Props {
  session: DemoSession;
  data: PersonalFeedState;
  industry: string | null;
  onFilter: (filter: FeedFilter) => void;
  onEvent: (eventId: string) => void;
  onProfile: () => void;
}

export function DashboardSummary({ session, data, industry, onFilter, onEvent, onProfile }: Props) {
  const summary = summarizeFeed(data.feed);
  const available = data.feed !== null;
  const value = (number: number) => available ? String(number).padStart(2, '0') : '—';
  const lead = summary.lead;
  const freshness = data.updatedAt ? new Date(data.updatedAt).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) : null;
  return <>
    <section className="command-deck" aria-label="Сводка персональной аналитики" aria-busy={data.loading}>
      <div className="deck-intro">
        <div className="deck-eyebrow"><span className="deck-live-dot"/> ПЕРСОНАЛЬНАЯ АНАЛИТИКА</div>
        <h1>Обзор<br/><em>событий</em></h1>
        <p>Отобранные новости и возможное<br/>влияние на компанию.</p>
        <button type="button" className="deck-company" onClick={onProfile}><span className="deck-company-avatar"><Icon name="briefcase" size={17}/></span><span><strong>{session.company_id} <span className="deck-role">{roles[session.role].short}</span></strong><small>{industry || 'Деятельность не указана'}</small></span><Icon name="chevron" size={15}/></button>
      </div>
      <div className="deck-agenda">
        <div className="deck-panel-label">Персональная лента</div>
        <button type="button" className="agenda-total" disabled={!available} onClick={() => onFilter('all')} aria-label="Показать все события персональной ленты"><strong data-testid="summary-total">{value(summary.total)}</strong><span>событий<br/>в ленте <Icon name="arrow" size={15}/></span></button>
        <div className="agenda-rows"><button type="button" disabled={!available} onClick={() => onFilter('critical')}><span><i className="agenda-dot dot-coral"/>Значимость 5/5</span><strong data-testid="summary-critical">{value(summary.critical)}</strong></button><button type="button" disabled={!available} onClick={() => onFilter('saved')}><span><i className="agenda-dot dot-lilac"/>С анализом</span><strong data-testid="summary-saved">{value(summary.saved)}</strong></button></div>
        <small className="agenda-scope">{data.error ? 'Не удалось обновить ленту' : freshness ? `Лента обновлена в ${freshness}` : data.loading ? 'Получаем персональную ленту…' : 'Данные пока недоступны'}</small>
      </div>
      <div className="deck-next">
        <div className="deck-panel-label"><Icon name="activity" size={15}/> Приоритетное событие</div>
        <span className="deck-next-tag">{lead?.relevance_grade != null ? `ЗНАЧИМОСТЬ ${lead.relevance_grade}/5` : 'ОЦЕНКА НЕ УКАЗАНА'}</span>
        <h2>{lead ? lead.insight?.headline || lead.title || 'Событие без заголовка' : !available ? data.loading ? 'Загружаем события' : 'Лента недоступна' : 'В ленте пока нет событий'}</h2>
        <p>{lead ? lead.relevance_assessment?.business_channel || 'Откройте событие, чтобы посмотреть доступные факты, выводы и источники.' : !available ? data.loading ? 'Ожидаем ответ сервера.' : 'Обновите ленту или повторите попытку позже.' : 'Запустите подбор ниже, чтобы проверить доступные публикации.'}</p>
        {summary.lead ? <button type="button" className="deck-next-button" onClick={() => onEvent(lead!.event_id)}>Открыть событие <Icon name="arrow" size={17}/></button> : <button type="button" className="deck-next-button" onClick={onProfile}>Профиль компании <Icon name="arrow" size={17}/></button>}
        <div className="deck-orb" aria-hidden="true"/>
      </div>
    </section>
    <section className="dashboard-widgets" aria-label="Индикаторы персональной ленты">
      <article className="summary-widget agenda-widget"><div className="widget-top"><span className="widget-icon"><Icon name="activity" size={18}/></span><h2>Значимость событий</h2><span className="widget-caption">ТЕКУЩАЯ ЛЕНТА</span></div><div className="pulse-numbers"><strong data-testid="pulse-critical">{value(summary.critical)}</strong><span>с максимальной<br/>значимостью</span><button type="button" className="widget-arrow" aria-label="Открыть события со значимостью 5 из 5" disabled={!available} onClick={() => onFilter('critical')}><Icon name="arrow" size={19}/></button></div><div className="pulse-bar" aria-hidden="true">{summary.total > 0 ? <>{summary.critical > 0 && <span className="pulse-critical" style={{ flex: summary.critical }}/>} {summary.high > 0 && <span className="pulse-high" style={{ flex: summary.high }}/>} {summary.other > 0 && <span className="pulse-other" style={{ flex: summary.other }}/>}</> : <span className="pulse-other"/>}</div><div className="pulse-legend"><span><i className="legend-critical"/>5/5 <b>{value(summary.critical)}</b></span><span><i className="legend-high"/>4/5 <b>{value(summary.high)}</b></span><span><i className="legend-other"/>Прочие <b>{value(summary.other)}</b></span></div></article>
      <article className="summary-widget bank-widget"><div className="widget-top"><span className="widget-icon"><Icon name="briefcase" size={18}/></span><h2>Позиция банка</h2><span className="widget-caption">МАТЕРИАЛЫ БАНКА</span></div><div className="bank-widget-main"><div className="bank-donut" style={{ background: `conic-gradient(#684dc4 ${summary.total ? summary.bank / summary.total * 360 : 0}deg, #ddd5f1 0deg)` }} aria-hidden="true"><span>{available ? summary.bank : '—'}</span></div><div><strong data-testid="summary-bank">{available ? `${summary.bank} из ${summary.total}` : '—'}</strong><p>событий с найденной<br/>позицией Альфа-Банка</p></div></div>{available && summary.historical > 0 && <span className="bank-history-note">Только историческая позиция: {summary.historical}</span>}<button type="button" className="widget-text-button" disabled={!available} onClick={() => onFilter('bank')}>Открыть события с позицией банка <Icon name="arrow" size={15}/></button></article>
      <article className="summary-widget action-widget"><div className="widget-top"><span className="widget-icon"><Icon name="check" size={18}/></span><h2>Что проверить</h2><span className="widget-caption">ПО СОБЫТИЯМ</span></div>{summary.actions.length ? <div className="widget-actions">{summary.actions.slice(0, 2).map((item, index) => <button type="button" key={item.event_id} onClick={() => onEvent(item.event_id)}><span className="action-index">{String(index + 1).padStart(2, '0')}</span><span>{item.relevance_assessment!.required_action}</span><Icon name="chevron" size={15}/></button>)}</div> : <p className="widget-empty">{available ? 'В текущей ленте нет конкретных вопросов для проверки. Откройте событие, чтобы изучить доступные факты.' : data.loading ? 'Загружаем вопросы для проверки…' : 'Вопросы появятся после загрузки ленты.'}</p>}<small>Применимость зависит от условий вашей компании.</small></article>
    </section>
    <p className="widget-scope-note"><Icon name="file" size={12}/> Показатели относятся к текущей персональной ленте, а не ко всем публикациям или оценке риска компании.</p>
  </>;
}
