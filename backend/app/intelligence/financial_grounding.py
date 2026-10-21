"""Shared business reasoning rules and a bounded check for a known false inference.

Rate transmission reference (general mechanism, not company exposure evidence):
https://www.cbr.ru/faq/dkp/01/ — «Как ЦБ влияет на инфляцию?».
"""
import re

BUSINESS_RULES = """
Бизнес-инсайт: событие → экономический механизм → условное последствие для бизнеса этой компании.
Объясни механизм в text, а в next_steps — какие данные проверить и какое решение от них зависит.
«Проверить договор» или «условия могут зависеть от ставки» без механизма и последствия — не инсайт.
Выбирай обоснованный канал: спрос клиентов, затраты, оборотный капитал, инвестиции или финансирование;
роль CFO и признак кредита не требуют выбирать долг. Неизвестную экспозицию явно обозначай гипотезой.
Общий механизм не обязан дословно повторяться в новости, но должен быть экономически обоснован;
это интерпретация AI, а не позиция банка. Условие «если» не оправдывает неверную причинность.
Высокая ставка обычно поддерживает сбережения и сдерживает кредитный спрос и инвестиции; сохранение
ставки не является её повышением. Реакция не мгновенна и не доказывает падение заказов этой компании.
Прогноз ставки не определяет неизменность условий кредита. Признак продукта не доказывает долг или свободные деньги;
без договора неизвестны тип ставки, привязка и пересмотр. Не переноси на них горизонт прогноза.
Из паузы по ключевой ставке не следует, что стоимость обслуживания не снизится: влияют остаток,
спред, комиссии и договор. ОКВЭД строительства не доказывает заёмное финансирование проектов.
Ставка на конец следующего года не доказывает, что снижение начнётся только тогда.
Разделяй уровень, изменение и прогноз: сохранение ставки не удорожает кредит само по себе;
пересмотр прогноза банком не является решением ЦБ или изменением договора. Ожидания могут влиять
на рыночные ставки только через отдельно объяснённый механизм, а не как свершившееся изменение.
Снижение темпа инфляции не означает падение уровня цен. ИПЦ не равен ценам материалов, топлива или зарплат
компании; переход к её затратам требует подтверждённой структуры закупок и условий индексации.
Выручка, прибыль и поступления денег — разные показатели. Перенос оплаты не меняет автоматически
признанную выручку; для денежного потока проверяй сроки оплат, для выручки — выполнение и условия признания.
Разбирай основное событие event.title. Старое решение, приведённое в статье как фон, не заменяет свежий факт.
Если обоснованного канала нет, оставь факты и missing_data; не сочиняй персональный эффект.
"""


def business_focus(registry, *, question=None, event_id=None, event_title=None):
    """Select a financial mechanism from the primary news and an evidenced industry."""
    if question:
        return None
    facts = {key: e for key, e in registry.items() if e.get("kind") == "event_fact"
             and (event_id is None or e.get("event_id") == event_id)}
    patterns = [('inflation', r'инфляц|индекс\w*\s+цен|\bипц\b'),
        ('policy_rate', r'ключев\w*\s+ставк|ставк\w*\s+цб'),
        ('fuel', r'топлив|бензин|дизел'), ('fx', r'курс\w*\s+(?:рубл|валют)|валютн\w*\s+курс'),
        ('labor', r'зарплат|оплат\w*\s+труд|кадр\w*\s+дефицит'),
        ('trade_rules', r'пошлин|таможен'), ('regulation', r'закон|регулирован|обязательн\w*\s+требован'),
        ('consumer_demand', r'потребительск\w*\s+спрос|спрос\w*\s+на\s+')]
    driver = next((topic for topic, pattern in patterns if re.search(pattern, event_title or '', re.I)), None)
    if driver is None:
        driver = next((topic for topic, _ in patterns if any(topic in e.get('attributes', {}).get('topics', [])
            and not re.match(r'\s*(?:как сообщалось|ранее\b)', e.get('text', ''), re.I) for e in facts.values())), None)
    industry = [key for key, e in registry.items() if e.get("kind") == "company_feature"
                and e.get("name") == "okved_code" and e.get('value')]
    if driver is None or not industry:
        return None
    news = [key for key, e in facts.items() if driver in e.get('attributes', {}).get('topics', [])]
    if event_title and not re.search(r'прогноз|ожидан|ожида', event_title, re.I):
        news = [key for key in news if facts[key].get('attributes', {}).get('kind') != 'forecast'
                and facts[key].get('attributes', {}).get('event_status') != 'forecast']
    title_words = ' '.join(re.findall(r'\w+', (event_title or '').casefold()))
    if len(title_words.split()) >= 4:
        exact = [key for key in news if title_words in
                 ' '.join(re.findall(r'\w+', facts[key].get('text', '').casefold()))]
        if exact:
            news = exact
    prefix = str(registry[industry[0]]['value']).split('.')[0]
    industry += [key for key, e in registry.items() if e.get("kind") == "company_feature"
                 and e.get("name") == "okved_name"]
    mechanisms = {
        'inflation': ('purchasing_costs', 'Если динамика подтверждённых закупочных цен меняется при фиксированной цене продажи, может меняться маржа.',
            'ИПЦ — общий фон: проверить цены конкретных ресурсов и условия индексации до изменения прогноза затрат.',
            ['Структура закупок и динамика цен ресурсов', 'Условия индексации договоров']),
        'fuel': ('operating_costs', 'Если топливо или подтверждённая транспортная составляющая существенны в себестоимости, изменение их цен может менять маржу.',
            'Проверить затратную базу и индексацию тарифов; наличие собственного автопарка не установлено.',
            ['Фактическая топливная и транспортная составляющая', 'Тарифы и условия индексации']),
        'fx': ('currency_pricing', 'Если закупки, продажи или цены ресурсов связаны с указанной валютой, изменение курса может влиять на маржу и платежи.',
            'Проверить валюту договоров и открытые позиции до расчёта эффекта; страна в ФТС не доказывает валюту расчётов.',
            ['Валюты и сроки договоров', 'Подтверждённые открытые позиции и условия изменения цен']),
        'labor': ('labor_costs', 'Если трудозатраты существенны, а цена заказа фиксирована, рост стоимости труда может снижать маржу выполнения.',
            'Сверить фактические трудозатраты, загрузку и модель ценообразования.',
            ['Трудозатраты и загрузка', 'Условия оплаты труда и цена договора']),
        'trade_rules': ('trade_compliance', 'Если товары и операции компании входят в область действия изменения, могут меняться затраты и требования к поставке.',
            'Сопоставить товар, юрисдикцию, стадию и дату вступления с конкретной операцией.',
            ['Товар и код ТН ВЭД', 'Юрисдикция, дата и применимость требования']),
        'regulation': ('operating_requirements', 'Если компания входит в область действия нормы, её выполнение может требовать изменения процессов и бюджета.',
            'Проверить применимость, статус нормы и срок; законопроект не является действующей обязанностью.',
            ['Статус и срок действия нормы', 'Подтверждённые критерии применимости к компании']),
        'consumer_demand': ('sales_demand', 'Если клиенты компании относятся к затронутому сегменту, изменение их спроса может менять объём продаж и загрузку.',
            'Сопоставить фактические заказы и сегменты клиентов с событием.',
            ['Сегменты клиентов и портфель заказов', 'Подтверждённые изменения спроса']),
    }
    if driver == 'policy_rate':
        channel, hypothesis, decision, unknowns = ('customer_investment_demand',
            'Если запуск проектов заказчиков зависит от кредита, длительная высокая ставка может откладывать запуск и поступления строительной компании.',
            'Надёжность плана заказов и поступлений; проверить финансирование и сроки запуска крупных заявок.',
            ['Зависимость проектов заказчиков от кредитов', 'Состав заказчиков и сроки запуска', 'Условия и сроки оплаты'])
        if prefix == '62':
            channel, hypothesis = 'customer_it_demand', 'Если заказчики финансируют ИТ-инвестиции кредитом, высокая стоимость финансирования может ограничивать бюджеты новых заказов разработки.'
        elif prefix in {'46', '47'}:
            channel, hypothesis = 'customer_demand_and_inventory', 'Если спрос покупателей чувствителен к кредитным условиям, дорогой кредит может замедлять продажи и оборот запасов.'
        elif prefix in {'49', '50', '51', '52'}:
            channel, hypothesis = 'logistics_demand', 'Если грузопотоки заказчиков зависят от финансируемых инвестиций и закупок, дорогой кредит может ограничивать спрос на перевозки.'
        elif prefix not in {'41', '42', '43'}:
            channel, hypothesis = 'investment_demand', 'Если покупатели финансируют инвестиционные закупки кредитом, высокая стоимость финансирования может сдерживать заказы компании.'
        if prefix not in {'41', '42', '43'}:
            decision = 'Сверить реальные заказы, сроки оплат и чувствительность клиентов к финансированию до изменения плана продаж и поступлений.'
            unknowns = ['Бизнес-модель и сегменты клиентов', 'Зависимость заказов от кредитных условий', 'Портфель заказов и сроки оплат']
    else:
        channel, hypothesis, decision, unknowns = mechanisms[driver]
    return {"channel": channel, 'driver': driver, "news_ids": news, "company_ids": industry,
            "max_inferences": 1 if news else 0, "hypothesis": hypothesis,
            "unknowns": unknowns + ([] if news else ['Нет проверенного факта об основном событии заголовка']),
            "decision": decision,
            "constraints": "Разбери этот канал, явно условно. Наличие долга самой компании не установлено. Разделяй прогноз, решение, уровень и изменение; старый фон не заменяет событие заголовка. При отсутствии основного факта оставь только факты и пробелы.",
            "not_inference_basis_ids": [key for key, e in registry.items() if e.get("name") == "CL_PRODUCT_LOA_MARK"]}


def focus_issues(draft, registry, *, question=None, event_id=None, event_title=None):
    focus = business_focus(registry, question=question, event_id=event_id, event_title=event_title)
    if not focus:
        return []
    issues = []
    if sum(claim.kind == "model_inference" for claim in draft.claims) > focus["max_inferences"]:
        issues.append("business_focus_requires_one_primary_inference")
    for claim in draft.claims:
        if claim.kind != "model_inference":
            continue
        if not set(claim.evidence_ids).intersection(focus["company_ids"]):
            issues.append(f"{claim.id}:business_focus_needs_industry_basis")
        if not set(claim.evidence_ids).intersection(focus['news_ids']):
            issues.append(f"{claim.id}:business_focus_needs_primary_news_basis")
        if set(claim.evidence_ids).intersection(focus["not_inference_basis_ids"]):
            issues.append(f"{claim.id}:credit_flag_does_not_establish_company_financing")
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
    for clause in re.split(r"[.!?;]|\b(?:но|однако)\b", " ".join(text.split()), flags=re.I):
        match = _PROJECTION.search(clause)
        if match and not _DISAVOWAL.search(clause[:match.start()]):
            return True
    return False
