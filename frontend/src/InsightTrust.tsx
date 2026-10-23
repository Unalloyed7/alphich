import { useEffect, useState } from 'react';
import type { AuditRun, Claim, Evidence, Insight } from './analysis';
import { analysisApi } from './api';
import { Icon } from './Icon';

export function InsightTrust({ insight, onDetails }: { insight: Insight; onDetails: (kind: Claim['kind']) => void }) {
  const groups = [{ kind: 'news_fact', label: 'Факты новости', icon: 'file' }, { kind: 'company_fact', label: 'О компании', icon: 'briefcase' }, { kind: 'bank_view', label: 'Позиция банка', icon: 'link' }, { kind: 'model_inference', label: 'Выводы модели', icon: 'activity' }] as const;
  const limited = insight.status === 'insufficient_evidence' || !!insight.verification?.issues?.length || insight.analysis_method_current === false;
  const verified = ['passed', 'verified', 'valid'].includes(insight.verification?.status);
  return <section className="trust-overview" aria-label="Состав и проверка анализа"><div className="trust-heading"><Icon name="shield" size={17}/><strong>На чём основан анализ</strong><span className={limited ? 'trust-limited' : ''}>{limited ? 'Есть ограничения' : verified ? 'Автоматическая проверка пройдена' : 'Статус проверки — в обосновании'}</span></div><div className="trust-parts">{groups.map(group => <button type="button" key={group.kind} onClick={() => onDetails(group.kind)}><Icon name={group.icon} size={16}/><span>{group.label}</span><strong>{insight.claims.filter(claim => claim.kind === group.kind).length}</strong></button>)}</div><p>{insight.missing_data.length > 0 && <><strong>Для применения вывода нужно уточнить данные ({insight.missing_data.length}).</strong> </>}Факты, позиция банка и выводы модели показаны отдельно. Проверка системой не заменяет экспертную оценку.</p></section>;
}

export function BankCitation({ insight, token }: { insight: Insight; token: string }) {
  const evidenceId = insight.claims.find(claim => claim.kind === 'bank_view')?.evidence_ids[0];
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    setEvidence(null); setError(false);
    if (!evidenceId) return;
    let active = true;
    void analysisApi.evidence(evidenceId, insight.id, token).then(value => { if (active) setEvidence(value); }).catch(() => { if (active) setError(true); });
    return () => { active = false; };
  }, [evidenceId, insight.id, token]);
  if (!evidenceId) return null;
  const date = evidence?.published_at ? new Date(evidence.published_at) : null;
  return <div className="bank-citation">{error ? <span>Не удалось загрузить дату материала. Источники доступны в обосновании.</span> : !evidence ? <span role="status">Уточняем источник позиции…</span> : <><strong>{evidence.title || 'Подтверждающий материал банка'}</strong><span>Дата материала: {date && !Number.isNaN(date.getTime()) ? date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }) : 'не указана'}{evidence.page != null ? ` · стр. ${evidence.page}` : ''}</span></>}</div>;
}

const actions: Record<string, string> = {
  select_business_focus: 'Определён бизнес-контекст события', bound_business_context: 'Уточнены границы доступных данных',
  get_company_features: 'Изучены подтверждённые признаки компании', get_event_facts: 'Прочитаны факты публикации',
  inspect_evidence: 'Проверены подтверждающие фрагменты', search_bank_materials: 'Выполнен поиск аналитики банка',
  search_bank: 'Выполнен поиск аналитики банка', search_related_events: 'Проверены связанные события',
  get_payment_neighbors: 'Изучены доступные платёжные связи', verify_card_schema: 'Проверена структура анализа',
  verify_card: 'Проверена поддержка выводов источниками', agent_handoff: 'Результаты переданы следующему этапу',
  extract_public_article: 'Извлечены факты публикации', grade_rechecked: 'Перепроверена значимость для компании', cache_hit: 'Использован сохранённый результат',
};
export function AuditTimeline({ run }: { run: AuditRun }) {
  const knownCosts = run.llm_calls.filter(call => call.cost_rub !== null && Number.isFinite(Number(call.cost_rub)));
  const total = knownCosts.reduce((sum, call) => sum + Number(call.cost_rub), 0);
  return <section className="audit-timeline" aria-label="Как подготовлен анализ"><h4>Как подготовлен анализ</h4><p>Показаны действия системы и доступные результаты проверки.</p><ol>{run.steps.map((step, index) => <li key={index}><span className="audit-index">{index + 1}</span><div><strong>{actions[step.action ?? ''] ?? 'Этап подготовки анализа'}</strong><details><summary>Технические сведения этапа</summary><pre>{JSON.stringify(step, null, 2)}</pre></details></div></li>)}</ol><div className="audit-costs"><span>Обращений к модели: {run.llm_calls.length}</span><span>Учтённая стоимость: {total.toLocaleString('ru-RU', { maximumFractionDigits: 4 })} ₽{knownCosts.length < run.llm_calls.length ? ' · часть расходов не подтверждена' : ''}</span></div></section>;
}
