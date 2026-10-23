import { Icon } from './Icon';
import { roles } from './roles';
import type { CompanyProfile, DemoSession } from './types';
import './profileOverview.css';

const products: [string, string][] = [
  ['CL_PRODUCT_RKO_MARK', 'Расчётное обслуживание'], ['CL_PRODUCT_GAR_MARK', 'Банковские гарантии'],
  ['CL_PRODUCT_LOA_MARK', 'Кредитные продукты'], ['CL_PRODUCT_LEA_MARK', 'Лизинг'],
  ['CL_PRODUCT_DEP_MARK', 'Депозиты'], ['CL_PRODUCT_ZPP_MARK', 'Зарплатный проект'], ['FACTORING_MARK', 'Факторинг'],
];
const limitations: Record<string, string> = {
  'This is an anonymized synthetic company profile.': 'Профиль компании обезличен.',
  'Revenue currency and snapshot date have not been confirmed.': 'Валюта выручки и дата снимка не подтверждены.',
  'Registration region is not proof of geographic operations.': 'Регион регистрации не определяет географию работы компании.',
  'FTS countries do not establish trade direction or transaction currency.': 'Страны в данных ФТС не подтверждают направление торговли и валюту расчётов.',
  'Credit flag does not disclose debt amount, contract rate, or maturity.': 'Кредитный признак не раскрывает сумму долга, ставку и сроки договора.',
};
function qualityLabel(flag: string) {
  if (flag === 'fts_country_without_flag') return 'Страны ФТС указаны без признака внешней торговли.';
  if (flag === 'fts_flag_without_country') return 'Есть признак ФТС, но страны не указаны.';
  if (flag.startsWith('unrecognized_boolean:')) return 'Один из признаков в источнике не удалось распознать.';
  return `Особенность исходных данных: ${flag}`;
}

export function ProfileOverview({ session, profile, error, onRetry, onSwitch }: {
  session: DemoSession; profile: CompanyProfile | null; error: string | null;
  onRetry: () => void; onSwitch: () => void;
}) {
  const role = roles[session.role];
  // Only render data belonging to the active company and perspective.
  const company = profile?.id === session.company_id && profile.role === session.role ? profile : null;
  const present = company ? products.filter(([code]) => company.products[code] === true) : [];
  const countryNames = company?.fts_countries?.filter(Boolean) ?? [];
  const clientType = company?.client_type === 'CLIENT' ? 'Клиент банка' : company?.client_type;
  return <div className="profile-overview">
    <section className="personal-profile" aria-labelledby="personal-profile-title">
      <div className="personal-profile-identity"><span className="personal-profile-avatar"><Icon name="briefcase" size={24}/></span><div><div className="eyebrow">ВАША РОЛЬ В КОМПАНИИ</div><h2 id="personal-profile-title">{role.title} <span>{role.short}</span></h2><p>Компания <strong>{session.company_id}</strong> · рабочий контекст аналитики</p></div><button type="button" className="secondary-button" onClick={onSwitch}>Сменить роль или компанию <Icon name="chevron" size={15}/></button></div>
      <div className="personal-profile-focus"><div><h3>{role.focus}</h3><p>{role.description}</p></div><ul aria-label="Приоритеты выбранной роли">{role.priorities.map(priority => <li key={priority}>{priority}</li>)}</ul></div>
      <p className="personal-profile-note">Роль учитывается при отборе событий и подготовке выводов. Права доступа проверяет сервер.</p>
    </section>

    <section className="company-passport panel" aria-labelledby="company-passport-title" aria-busy={!company && !error}>
      <div className="company-passport-heading"><div><div className="eyebrow">КОМПАНИЯ В ЦИФРАХ И ФАКТАХ</div><h2 id="company-passport-title">Ваш бизнес в контексте</h2><p>Данные, на которые опирается персональная аналитика.</p></div><span className="company-id">{session.company_id}</span></div>
      {error ? <div className="profile-load-error"><p role="alert">Не удалось загрузить профиль компании: {error}</p><button type="button" className="secondary-button" onClick={onRetry}>Повторить загрузку профиля <Icon name="refresh" size={15}/></button></div> : !company ? <p className="profile-loading" role="status"><span className="spinner tiny"/> Загружаем данные компании…</p> : <>
        <dl className="company-passport-facts"><div className="company-fact-industry"><dt>Основная деятельность</dt><dd><span>{company.industry.okved_name || 'Не указана'}</span><small>ОКВЭД {company.industry.okved_code || 'не указан'}</small></dd></div><div><dt>Регион регистрации</dt><dd><span>{company.region_registration || 'Не указан'}</span><small>{company.city_registration || 'Город не указан'}</small></dd></div><div><dt>Выручка из источника</dt><dd><span>{company.revenue_raw || 'Нет данных'}</span><small>{company.revenue_unit || 'Единица измерения не подтверждена'}</small></dd></div><div><dt>Тип клиента</dt><dd><span>{clientType || 'Не указан'}</span><small>Значение из профиля компании</small></dd></div></dl>
        <div className="company-context-grid"><section aria-labelledby="profile-products-title"><h3 id="profile-products-title">Банковские продукты</h3><p>{present.length ? 'В исходном профиле отмечены:' : 'В исходном профиле нет отмеченных продуктов.'}</p><div className="profile-product-list">{present.map(([code, title]) => <span key={code}><Icon name="check" size={14}/>{title}</span>)}</div><details className="profile-product-details"><summary>Все признаки продуктов</summary><dl>{products.map(([code, title]) => <div key={code}><dt>{title}</dt><dd>{company.products[code] === true ? 'Есть признак' : company.products[code] === false ? 'Не отмечено в профиле' : 'Нет данных'}</dd></div>)}</dl></details><small>Признак продукта не раскрывает условия договора.</small></section><section aria-labelledby="profile-trade-title"><h3 id="profile-trade-title">Внешнеторговый контекст</h3><p>{company.fts_flag === true ? 'Есть признак в данных ФТС' : company.fts_flag === false ? 'Признак ФТС не отмечен' : 'Нет данных о признаке ФТС'}</p>{countryNames.length ? <ul className="profile-country-list" aria-label="Страны в данных ФТС">{countryNames.map((country, index) => <li key={`${country}:${index}`}>{country}</li>)}</ul> : <p className="profile-secondary">Подтверждённого списка стран нет.</p>}<small>Страны не устанавливают направление торговли и валюту операций.</small></section></div>
        {company.quality_flags.length > 0 && <div className="profile-quality"><Icon name="alert" size={17}/><div><strong>В данных есть особенности</strong><ul>{[...new Set(company.quality_flags.map(qualityLabel))].map(flag => <li key={flag}>{flag}</li>)}</ul></div></div>}
        <details className="profile-provenance"><summary>Источники и ограничения профиля</summary><p>Версия данных: <span className="profile-version">{company.profile_version}</span></p><p>Признаков с привязкой к источникам: {company.features.length}. Наличие признака само по себе не подтверждает влияние новости на компанию.</p><ul>{company.limitations.map((item, index) => <li key={index}>{limitations[item] ?? item}</li>)}</ul></details>
      </>}
    </section>
    <section className="personalization-guide" aria-labelledby="personalization-guide-title"><h2 id="personalization-guide-title">Как складывается ваш контекст</h2><ol><li><span>01</span><div><h3>Компания</h3><p>Отрасль, признаки продуктов и другие доступные данные помогают найти связь события с бизнесом.</p></div></li><li><span>02</span><div><h3>Ваша роль</h3><p>{role.description} Этот фокус помогает определить, на какие последствия обратить внимание.</p></div></li><li><span>03</span><div><h3>Подтверждения</h3><p>Факты, материалы банка и ограничения позволяют проверить, на чём основан каждый вывод.</p></div></li></ol></section>
  </div>;
}
