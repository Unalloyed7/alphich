"""Observed company inputs and testable hypotheses; no invented business exposures."""
from ..common import digest
from .inputs import industry_matches
from .profiles import normalize_company
from .payments import METHOD, validate_payment_feature

VERSION = "business-context-v2"

# Activity suggests questions; it does not prove a problem or a business model.
LENSES = {
    "41": [
        ("customer_investment", "Если заказчики откладывают инвестиционные проекты, может меняться спрос на строительные работы.",
         "Инвестиционные решения заказчиков → сроки заказов → загрузка строительных мощностей.",
         "Портфель заказов, стадии проектов и переносы решений заказчиков.", "Планировать загрузку по подтверждённому портфелю."),
        ("project_costs", "Если ресурсы дорожают быстрее договорной цены, под давлением может оказаться маржа проектов.",
         "Цены ресурсов и индексация → сметная себестоимость → маржа конкретного проекта.",
         "Сметы, закупаемые материалы, фиксированные цены и условия индексации.", "Проверять сметы и условия новых предложений."),
    ],
    "62": [
        ("customer_it_budgets", "Если заказчики сокращают ИТ-бюджеты, новые проекты разработки могут переноситься или уменьшаться.",
         "ИТ-бюджеты заказчиков → объём и сроки заказа разработки → загрузка команды и выручка.",
         "Заказная или продуктовая модель, отрасли заказчиков, воронка и подтверждённые переносы.",
         "Уточнять прогноз продаж по фактическим решениям заказчиков."),
        ("delivery_economics", "Если трудозатраты растут при фиксированной цене разработки, может снижаться маржа заказа.",
         "Трудозатраты и ставка команды → стоимость выполнения заказа → маржа при заданной цене.",
         "Модель оплаты, оценки трудозатрат, загрузка команды и фактическая себестоимость.",
         "Пересматривать оценки и условия новых заказов по фактическим трудозатратам."),
    ],
    "46": [
        ("trading_margin", "Если закупочные цены растут быстрее отпускных, может снижаться торговая маржа.",
         "Закупочные цены и условия поставок → стоимость товара → маржа при заданной цене продажи.",
         "Ассортимент, валюты закупок, прайс-листы, сроки поставок и возможность менять отпускные цены.",
         "Уточнять цены и условия пополнения запасов по подтверждённому ассортименту."),
        ("inventory_cash_cycle", "Если товар продаётся медленнее срока оплаты поставщикам, может увеличиваться потребность в оборотных средствах.",
         "Оборачиваемость запасов и сроки расчётов → разрыв между выплатами и поступлениями.",
         "Возраст запасов, сроки оплаты, авансы, дебиторская и кредиторская задолженность.",
         "Согласовывать пополнение запасов и платёжный план по фактическим срокам."),
    ],
    "49": [
        ("transport_costs", "Если топливо и другие затраты растут быстрее тарифов, может снижаться маржа перевозок.",
         "Затраты на выполнение перевозки → себестоимость рейса → маржа при договорном тарифе.",
         "Вид транспорта, маршруты, топливная составляющая и условия индексации тарифов.",
         "Проверять тарифы и экономику рейсов по фактическим маршрутам."),
        ("transport_demand", "Если заказчики сокращают перевозки, может снижаться загрузка транспортных мощностей.",
         "Грузопотоки заказчиков → загрузка → распределение постоянных затрат на выполненные перевозки.",
         "Грузы, маршруты, заказы и фактическая загрузка; состав и владение парком неизвестны.",
         "Корректировать план загрузки после подтверждения изменения заказов."),
    ],
}

INDUSTRIAL_LENSES = [
    ("production_margin", "Если дорожают используемые ресурсы без изменения цены продукции, может снижаться производственная маржа.",
     "Цены подтверждённых ресурсов → себестоимость продукции → маржа при заданной цене реализации.",
     "Номенклатура, ресурсная спецификация, энергоёмкость и условия индексации договоров.",
     "Пересматривать себестоимость и условия новых заказов по фактической структуре затрат."),
    ("production_load", "Если меняется спрос на продукцию компании, может измениться загрузка мощностей.",
     "Подтверждённые заказы → выпуск и загрузка → распределение постоянных затрат.",
     "Продуктовые рынки, портфель заказов, мощности, запасы и возможность менять производственный план.",
     "Сопоставлять производственный план с подтверждённым портфелем заказов."),
]


def business_context(profile):
    company = normalize_company(profile)
    observed, hypotheses = [], []

    def observe(code, value, ids, meaning):
        if ids:
            observed.append({"code": code, "value": value, "evidence_ids": sorted(ids), "meaning": meaning})

    observe("industry", company["industry"], company["bindings"]["industry"],
            "Основная деятельность из профиля; продуктовый сегмент и бизнес-модель не установлены.")
    observe("registration", company["region"], company["bindings"]["region"],
            "Место регистрации; не подтверждает географию операций, проектов или клиентов.")
    for product in ("loan", "deposit", "lease", "factoring"):
        if company["products"][product] is True:
            observe(product + "_flag", True, company["bindings"][product],
                    "Продукт указан в профиле; остатки, договорные условия и текущая экспозиция неизвестны.")
    if company["products"]["loan"] is True and company["bindings"]["loan"]:
        hypotheses.append({"id": "financing_sensitivity", "status": "requires_confirmation",
            "basis_feature_ids": company["bindings"]["loan"],
            "possible_problem": "Если есть действующий долг с плавающей ставкой или близким рефинансированием, условия финансирования могут влиять на денежный поток.",
            "mechanism": "Подтверждённый остаток и условия договора → процентные платежи и даты выплат → платёжный план.",
            "check_data": "Остаток долга, фиксированная или плавающая ставка, формула, даты пересмотра, погашения и рефинансирования.",
            "decision": "Рассчитать сценарии платежей по договору; прогноз ключевой ставки сам по себе не изменяет договор."})
    if company["fts_flag"] is True:
        observe("customs_countries", company["countries"], company["bindings"]["fts"],
                "Страны в сведениях ФТС; направление, товары, валюты и действующие сделки не подтверждены.")
    for f in company["features"].values():
        if f["name"] == "revenue_raw" and f["value"] is not None:
            observe("revenue_raw", f["value"], [f["id"]],
                "Исходное значение выручки. Единица, валюта и период не подтверждены; денежный масштаб и эффект по нему не рассчитываются.")
        if f["name"] == "group_id" and f["value"]:
            observe("group_membership", f["value"], [f["id"]],
                    "Общий GROUP_ID; не доказывает контроль, консолидированный бюджет, гарантию или общий долг.")
        if f["name"] not in {"payment_incoming", "payment_outgoing"}:
            continue
        if f.get("method") != METHOD:
            continue
        validate_payment_feature(f)
        v = f["value"]
        observe(f["name"], {k: v.get(k) for k in ("payment_count", "counterparty_count", "top_counterparty_share",
                "known_profile_amount_share", "within_known_group_amount_share", "known_sectors")}, [f["id"]],
                "Структура наблюдаемого платёжного потока; не состав клиентов или поставщиков. Период, валюта и назначение неизвестны.")
        if v.get("top_counterparty_share") is not None and v["top_counterparty_share"] >= .5:
            hypotheses.append({"id": f["name"] + "_concentration", "status": "requires_confirmation",
                "basis_feature_ids": [f["id"]], "possible_problem": "Если крупнейший поток отражает регулярную операционную связь, его концентрация может быть существенна для бизнеса.",
                "mechanism": "Концентрация наблюдаемого потока → возможная чувствительность к изменению одной связи.",
                "check_data": "Период и полнота платежей, назначение, разовые операции и хозяйственная роль контрагента.",
                "decision": "Проверить устойчивость потоков после установления их хозяйственного смысла."})
    all_lenses = dict(LENSES)
    for construction in ("42", "43"):
        all_lenses[construction] = LENSES["41"]
    all_lenses["47"] = LENSES["46"]
    for transport in ("50", "51", "52"):
        all_lenses[transport] = LENSES["49"]
    code = company["industry"]["code"] or ""
    if code.split('.')[0].isdigit() and 10 <= int(code.split('.')[0]) <= 33:
        all_lenses[code.split('.')[0]] = INDUSTRIAL_LENSES
    for prefix, lenses in all_lenses.items():
        if industry_matches(company["industry"]["code"], prefix) and company["bindings"]["industry"]:
            for code, problem, mechanism, check, decision in lenses:
                hypotheses.append({"id": code, "status": "requires_confirmation",
                    "basis_feature_ids": company["bindings"]["industry"], "possible_problem": problem,
                    "mechanism": mechanism, "check_data": check, "decision": decision})
    gaps = [
        {"id": "business_model", "question": "Какие товары или услуги компания продаёт и кому?"},
        {"id": "operations", "question": "Где фактически находятся операции, проекты и клиенты?"},
        {"id": "contracts", "question": "Как устроены цены, индексация, оплата и обязательства по договорам?"},
        {"id": "financial_snapshot", "question": "Каковы дата, единицы, валюта и сопоставимый период финансовых данных?"},
        {"id": "eligibility", "question": "Какие лицензии, аккредитации, членства и специальные режимы подтверждены?"},
    ]
    if not any(o["code"].startswith("payment_") for o in observed):
        gaps.append({"id": "payment_context", "question": "Есть ли проверенный снимок платёжных связей этой компании?"})
    priorities = (["financial_snapshot", "contracts", "payment_context", "business_model", "operations", "eligibility"]
                  if company["role"] == "cfo" else
                  ["business_model", "operations", "contracts", "financial_snapshot", "payment_context", "eligibility"])
    purposes = {"business_model": "Определить рынки сбыта и проверить отраслевую применимость новости.",
        "operations": "Проверить географию влияния, маршруты, проекты и зависимости.",
        "contracts": "Разделить риск маржи, условия оплаты и движение денег.",
        "financial_snapshot": "Оценить масштаб и рассчитать эффект на сопоставимых подтверждённых данных.",
        "eligibility": "Проверить, распространяется ли требование или льгота на это юрлицо.",
        "payment_context": "Проверить концентрацию и назначение наблюдаемых потоков."}
    gaps.sort(key=lambda g: priorities.index(g["id"]))
    for gap in gaps:
        gap["purpose"] = purposes[gap["id"]]
    result = {"version": VERSION, "company_id": company["id"], "profile_version": company["profile_version"],
        "observed": observed, "hypotheses": hypotheses, "missing_data": gaps,
        "market_position": {"market_share": None, "status": "unknown",
            "reason": "Нет сопоставимых выручки, периода и полного отраслевого знаменателя; доля в выборке не является долей рынка."},
        "group_policy": "Анализ выбранного юрлица. Передача эффекта внутри группы требует отдельного основания; чужие признаки и обязательства не наследуются.",
        "limitations": sorted(set(company["limitations"] + ["hypotheses_are_not_company_facts", "external_identity_not_established"]))}
    result["context_hash"] = digest(result)
    return result
