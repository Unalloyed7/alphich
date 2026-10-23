import { useEffect, useState } from 'react';
import { analysisApi } from './api';
import type { DemoSession } from './types';
import './companyAnalysis.css';
import { Icon } from './Icon';

export interface BusinessContext {
  company_id: string; profile_version: string; context_hash: string;
  observed: Array<{ code: string; value: unknown; evidence_ids: string[]; meaning: string }>;
  hypotheses: Array<{ id: string; status: string; basis_feature_ids: string[]; possible_problem: string;
    mechanism: string; check_data: string; decision: string }>;
  missing_data: Array<{ id: string; question: string; purpose: string }>;
  factor_checks?: Array<{ factor: string; check: string }>;
  market_position: { market_share: number | null; status: string; reason: string };
  group_policy: string;
}

const labels: Record<string, string> = {
  industry: 'Деятельность', registration: 'Регистрация', loan_flag: 'Кредитный продукт в профиле',
  deposit_flag: 'Депозитный продукт в профиле', lease_flag: 'Лизинг в профиле', factoring_flag: 'Факторинг в профиле',
  revenue_raw: 'Исходная выручка', customs_countries: 'Страны в данных ФТС', group_membership: 'Группа компаний',
  payment_incoming: 'Наблюдаемые входящие платежи', payment_outgoing: 'Наблюдаемые исходящие платежи',
  organization_type: 'Тип организации', ownership_form: 'Форма собственности',
  company_age_band: 'Возраст в исходном снимке', business_stage_label: 'Исходная метка стадии бизнеса',
  negative_mark_raw: 'Признак без подтверждённой расшифровки',
  employees_fns_band: 'Численность ФНС', employees_spark_band: 'Численность СПАРК', employees_band: 'Согласованный диапазон численности',
  city_registration: 'Город регистрации', client_type: 'Отношения с банком',
  guarantee_flag: 'Гарантийный продукт в профиле', rko_flag: 'Расчётное обслуживание', payroll_flag: 'Зарплатный проект',
};

function valueText(value: unknown): string {
  if (value === true) return 'Есть признак';
  if (value === false) return 'Признак отсутствует';
  if (value === null || value === undefined) return 'Не установлено';
  if (Array.isArray(value)) return value.map(String).join(', ') || 'Не указано';
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>;
    if ('code' in record && 'name' in record) return [record.code, record.name].filter(Boolean).join(' · ');
    const parts: string[] = [];
    if (record.counterparty_count !== null && record.counterparty_count !== undefined) parts.push(`Контрагентов: ${record.counterparty_count}`);
    if (typeof record.top_counterparty_share === 'number') parts.push(`Доля крупнейшего: ${(record.top_counterparty_share * 100).toFixed(1)}%`);
    if (typeof record.known_profile_amount_share === 'number') parts.push(`Покрытие профилями: ${(record.known_profile_amount_share * 100).toFixed(1)}%`);
    return parts.join(' · ') || 'Наблюдаемый платёжный поток';
  }
  return String(value);
}

export function CompanyAnalysis({ session }: { session: DemoSession }) {
  const [context, setContext] = useState<BusinessContext | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setContext(null); setError(null);
    void analysisApi.businessContext(session.company_id, session.token).then(value => { if (alive) setContext(value); })
      .catch(e => { if (alive) setError(e instanceof Error ? e.message : 'Не удалось загрузить анализ компании'); });
    return () => { alive = false; };
  }, [session.company_id, session.token]);
  if (error) return <section className="panel company-analysis"><p role="alert">{error}</p></section>;
  if (!context) return <section className="panel loading-panel">Загружаем анализ профиля компании…</section>;
  return <section className="panel company-analysis" aria-label="Анализ профиля компании">
    <div className="panel-heading"><div><div className="eyebrow">АНАЛИЗ ПРОФИЛЯ</div><h2>Факторы влияния на компанию</h2>
      <p>Факты из профиля и гипотезы для проверки. Эти гипотезы используются при подготовке новостного анализа.</p></div></div>
    <details className="business-observed-details"><summary>Подтверждённые признаки и их значение · {context.observed.length}</summary>
    <div className="business-observed">{context.observed.map(item => <article key={item.code}>
      <span>{labels[item.code] ?? item.code}</span><strong>{valueText(item.value)}</strong><details><summary>Что это означает</summary><p>{item.meaning}</p></details>
    </article>)}</div></details>
    {!!context.factor_checks?.length && <details className="business-missing"><summary>Проверки применимости новостей · {context.factor_checks.length}</summary>
      <p>Совпадение признака помогает выбрать вопрос для проверки. Оно не означает, что требование или программа уже применимы к компании.</p>
      <ol className="business-questions">{context.factor_checks.map(item => <li key={item.factor}><strong>{labels[item.factor] ?? item.factor}</strong><p>{item.check}</p></li>)}</ol>
    </details>}
    <h3>Гипотезы для проверки</h3>
    <p className="muted">Применимость к этой компании требует подтверждения. Гипотеза не означает, что проблема уже возникла.</p>
    {context.hypotheses.length === 0 && <p>Для подготовки гипотез недостаточно данных. Вопросы для уточнения приведены ниже.</p>}
    <div className="business-hypotheses">{context.hypotheses.map(item => <article key={item.id}>
      <span className="business-hypothesis-badge">Гипотеза · нужно подтвердить</span><h4>{item.possible_problem}</h4>
      <details className="hypothesis-details"><summary>Механизм влияния и возможное решение <Icon name="chevron" size={15}/></summary><dl><dt>Как передаётся влияние</dt><dd>{item.mechanism}</dd><dt>Какие данные запросить</dt><dd>{item.check_data}</dd>
        <dt>Какое решение от них зависит</dt><dd>{item.decision}</dd></dl>
      <p className="business-basis">Основание в профиле: {context.observed.filter(o => o.evidence_ids.some(id => item.basis_feature_ids.includes(id)))
        .map(o => `${labels[o.code] ?? o.code}: ${valueText(o.value)}`).join('; ')}</p></details>
    </article>)}</div>
    <details className="business-missing"><summary>Вопросы для уточнения · {context.missing_data.length}</summary><h3>Что уточнить {session.role === 'cfo' ? 'финансовому директору' : 'руководителю'}</h3>
    <ol className="business-questions">{context.missing_data.map(item => <li key={item.id}><strong>{item.question}</strong><p>{item.purpose}</p></li>)}</ol></details>
    <div className="business-boundaries"><h3>Положение на рынке и границы анализа</h3><p>{context.market_position.reason}</p><p>{context.group_policy}</p>
      <p>Денежный эффект не рассчитан: нужны подтверждённые объёмы, период, валюта и условия договоров.</p></div>
  </section>;
}
