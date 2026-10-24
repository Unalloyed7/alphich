"""Open business-impact guidance and a bounded known-financing-fallacy check.

The historical module name is kept for callers. Neither topics nor financial
mechanisms determine whether an evidenced business consequence may be analyzed.
"""
import re

BUSINESS_RULES = """
Бизнес-инсайт: событие → объяснимый механизм влияния → условное последствие именно для этой компании.
Влияние может касаться репутации и доверия, права и соблюдения требований, безопасности людей и данных,
непрерывности работы, технологий, сотрудников, клиентов, конкуренции, окружающей среды, финансов
или другого существенного аспекта бизнеса. Это примеры, а не закрытая классификация или обязательные темы.
Не своди любой эффект к деньгам, выручке или долгу. Репутационное, правовое или техническое последствие
может быть самостоятельным; денежная оценка для него не обязательна и без данных недопустима.
Определи возможные последствия по основному событию и подтверждённому профилю, сравни их основания
и выбери наиболее полезные для решения пользователя. Не следуй заранее выбранному сервером каналу.
Для каждого вывода свяжи проверенный факт новости с содержательным признаком компании: лицензией,
продуктом, клиентами, процессом, брендом, площадкой или любым другим подходящим доказательством.
ОКВЭД не обязателен и сам по себе не подтверждает конкретную экспозицию. Неизвестную экспозицию
обозначай вопросом или гипотезой; сходство отраслей не доказывает инцидент, нарушение или ущерб компании.
Объясни механизм в text, а в next_steps — какие данные проверить, кто отвечает и какое решение зависит
от результата. Если связь не подтверждается, не назначай компании чужой риск и не объявляй нарушение.
Запрос данных сам по себе не является последствием новости. Не выдумывай эффект ради персонализации.
Общий причинный механизм может быть интерпретацией AI, но должен быть обоснован; не приписывай его
банку без отдельного банковского тезиса. Условие «если» не исправляет неверную причинность.
Сохраняй исходные признаки в контексте и различай факт, допущение и неизвестное. Банковский продукт
не доказывает действующий долг, остатки, свободные деньги или условия договора. Подтверждённые сведения
о договоре могут использоваться; кредитный флаг не должен скрывать их или заранее выбирать основной эффект.
Для событий о ставке разделяй уровень, изменение и прогноз: сохранение ставки не является повышением,
а прогноз банка не меняет договор. Высокая ставка обычно сдерживает кредитный спрос и инвестиции;
это не доказывает падение заказов конкретной компании. Без договора неизвестны тип ставки, привязка,
пересмотр, остаток, спред и комиссии. Горизонт прогноза не устанавливает неизменность условий кредита.
ОКВЭД строительства не доказывает заёмное финансирование проектов. Ставка на конец следующего года
не доказывает, что снижение начнётся только тогда. Ожидания и фактические решения — разные события.
Снижение темпа инфляции не означает падение уровня цен; ИПЦ не равен затратам компании.
Выручка, прибыль и поступления денег различаются; перенос оплаты не меняет автоматически выручку.
Место регистрации не устанавливает географию операций, а чужой инцидент — собственный инцидент компании.
Численность ФНС/СПАРК — диапазон с возможным неизвестным периодом, не текущий штат, зарплатный бюджет
или производительность. Возраст и непрозрачная метка стадии не доказывают рост или кризис.
Отрицательная метка без расшифровки не доказывает банкротство, нарушение, плохую репутацию или платёжеспособность.
Клиентский тип банка не определяет клиентов самой компании. Группа компаний не переносит автоматически
обязанности, санкции или инциденты между юрлицами; нужны подтверждённые связи и применимость.
Разбирай основное событие event.title. Историческое решение или прогноз из фона не заменяет свежий факт.
Если обоснованного влияния нет, оставь проверенные факты и missing_data, а не персональный эффект.
"""


_OPAQUE_ONLY_BASIS = {
    'company_age_band', 'business_stage_label', 'negative_mark_raw', 'client_type', 'group_id',
}
_BACKGROUND = re.compile(r'\s*(?:как сообщалось|ранее\b)', re.I)


def _company_lenses(registry):
    """Preserve source meanings without selecting or vetoing an impact domain."""
    groups = [
        ({'revenue_raw', 'revenue_unit', 'revenue_currency', 'revenue_period', 'revenue_scope'},
         'Отчётный масштаб с единицами, периодом и охватом; не прибыль, ликвидность или эффект новости.'),
        ({'employees_band', 'employees_fns_band', 'employees_spark_band'},
         'Диапазоны численности с проверкой периода и сопоставимости. Не текущий штат, зарплаты или производительность.'),
        ({'region_registration', 'city_registration'},
         'Только место регистрации; регион операций, площадки, аудитория и маршруты не установлены.'),
        ({'organization_type', 'ownership_form'},
         'Исходные правовые признаки; применимость конкретной нормы или программы нужно проверять отдельно.'),
        ({'company_age_band', 'business_stage_label', 'negative_mark_raw', 'client_type', 'group_id'},
         'Описательные или нерасшифрованные метки, не самостоятельное доказательство роста, кризиса, репутации или чужой экспозиции.'),
        ({'payment_incoming', 'payment_outgoing'},
         'Наблюдаемые платежи не устанавливают задолженность, просрочку, договорные условия или роль контрагента.'),
        ({'CL_PRODUCT_LOA_MARK', 'CL_PRODUCT_DEP_MARK', 'CL_PRODUCT_SAL_MARK'},
         'Признаки банковских продуктов; не действующий долг, остатки, свободные средства, зарплатный бюджет или условия договора.'),
    ]
    result = []
    for names, meaning in groups:
        ids = [key for key, item in registry.items() if item.get('kind') == 'company_feature'
               and item.get('name') in names and item.get('value') is not None and item.get('value') != '']
        if ids:
            result.append({'company_ids': ids, 'meaning': meaning})
    return result


def business_focus(registry, *, question=None, event_id=None, event_title=None):
    """Provide open impact selection guidance; do not preselect a mechanism.

    Every non-background fact in the current event is available regardless of its
    topic. The semantic reviewer must check which fact actually supports the
    headline and each consequence; a word/industry whitelist cannot prove that.
    """
    if question:
        return None
    facts = {key: item for key, item in registry.items() if item.get('kind') == 'event_fact'
             and (event_id is None or item.get('event_id') == event_id)}
    company = [key for key, item in registry.items() if item.get('kind') == 'company_feature']
    if not facts or not company:
        return None
    news = [key for key, item in facts.items() if not _BACKGROUND.match(item.get('text', ''))]
    industry = [key for key in company if registry[key].get('name') in {'okved_code', 'okved_name'}]
    return {
        'mode': 'open_business_impact_v2',
        'channel': 'open_business_impact',
        'driver': 'open',
        'event_title': event_title,
        'news_ids': news,
        'company_ids': company,
        'industry_basis_ids': industry,
        'company_context': _company_lenses(registry),
        'max_inferences': 2 if news else 0,
        'hypothesis': 'Выведи возможные последствия из основного события и профиля компании; заранее выбранного канала нет.',
        'unknowns': ['Если экспозиция или применимость не подтверждены, укажи, какой факт отсутствует; не объявляй их установленными.'],
        'decision': 'Выбери наиболее обоснованное влияние и конкретное действие, соответствующее последствиям и роли пользователя.',
        'decision_checks': [{
            'data': 'Какие доказательства связывают основное событие с этой компанией и какие данные ещё нужны?',
            'decision_branch': 'При подтверждении связи предложи соответствующее действие; при её отсутствии не переноси на компанию чужое событие.',
        }],
        'selection_policy': {
            'impact_domains_open': True,
            'industry_evidence_required': False,
            'choose_by': 'primary_event_company_exposure_and_decision_value',
            'bank_evidence': 'optional_and_separately_attributed',
        },
        'constraints': 'Не своди последствия к финансам. Выбери влияние по доказательствам, а не по ключевому слову или ОКВЭД. '
                       'Старый фон не заменяет основное событие. Неизвестное не установлено; при отсутствии основного факта оставь факты и пробелы.',
        'not_inference_basis_ids': [],
    }


def focus_issues(draft, registry, *, question=None, event_id=None, event_title=None):
    """Check provenance, never reject an impact because of its subject or industry."""
    if question:
        return []
    focus = business_focus(registry, event_id=event_id, event_title=event_title)
    primary_news_ids = set(focus['news_ids'] if focus else [
        key for key, item in registry.items() if item.get('kind') == 'event_fact'
        and (event_id is None or item.get('event_id') == event_id)
        and not _BACKGROUND.match(item.get('text', ''))])
    issues = []
    inferences = [claim for claim in draft.claims if claim.kind == 'model_inference']
    if focus and len(inferences) > focus['max_inferences']:
        issues.append('business_focus_requires_bounded_primary_inferences')
    for claim in inferences:
        company = [registry[key] for key in claim.evidence_ids if key in registry
                   and registry[key].get('kind') == 'company_feature']
        if company and all(item.get('name') in _OPAQUE_ONLY_BASIS for item in company):
            issues.append(f'{claim.id}:business_inference_needs_operating_basis')
        if not set(claim.evidence_ids).intersection(primary_news_ids):
            issues.append(f'{claim.id}:business_focus_needs_primary_news_basis')
    return issues


_FINANCING = (r"(?:стоимост\w*\s+обслуживан\w*|кредитн\w*\s+услови\w*|услови\w*\s+(?:\w+\s+){0,3}(?:финансирован\w*|кредит\w*|договор\w*)"
              r"|(?:стоимост\w*|ставк\w*|плат[её]ж\w*|нагрузк\w*)\s+(?:\w+\s+){0,3}"
              r"(?:кредит\w*|долг\w*|займ\w*))")
_STABILITY = (r"(?:сохранятся|сохранится|сохраняться|сохраняются|сохраняется|останутся|останется"
              r"|оставаться|остаются|остаётся|остается|неизменн\w*|стабильн\w*"
              r"|не\s+(?:изменятся|изменится|меняются|меняется|поменяются|поменяется|снизится|снизятся|снижается|снизиться))")
_PROJECTION = re.compile(rf"{_FINANCING}.{{0,100}}?{_STABILITY}|{_STABILITY}.{{0,80}}?{_FINANCING}", re.I)
_DISAVOWAL = re.compile(
    r"не\s+означа\w*|не\s+(?:позволя\w*|да[её]т).{0,45}(?:утвержда\w*|вывод\w*|основан\w*)"
    r"|нельзя\s+(?:утвержда\w*|счита\w*|вывод\w*)|неизвестно"
    r"|(?:проверить|уточнить).{0,35}(?:ли\b|будут\s+ли)", re.I)


def unsupported_financing_stability(text):
    """Reject positive stability predictions; keep explicit denials/unknowns.

    This targets a known failure pattern, not a general natural-language proof.
    The independent semantic review is still required for other implications.
    """
    # A metaphor about a company's reputation is not a financial contract.
    text = re.sub(r'кредит\w*\s+довер\w*', 'доверие', text, flags=re.I)
    for clause in re.split(r"[.!?;]|\b(?:но|однако)\b", " ".join(text.split()), flags=re.I):
        match = _PROJECTION.search(clause)
        if match and not _DISAVOWAL.search(clause[:match.start()]):
            return True
    return False
