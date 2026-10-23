"""How source observations may inform a news check; none are impact estimates."""

FACTOR_POLICIES = {
    "organization_type": {
        "meaning": "Тип организации из исходника; проверяется при наличии такого условия в новости. Не подтверждает лицензию, льготу или участие в закупке.",
        "topics": ["regulation", "trade_rules"],
        "check": "Сопоставить точный круг организаций в требовании или программе с подтверждённым типом организации.",
        "limitations": ["legal_eligibility_requires_explicit_event_conditions"],
    },
    "ownership_form": {
        "meaning": "Форма собственности из исходника; сама по себе не устанавливает государственные контракты, субсидии или гарантии.",
        "topics": ["regulation", "capex"],
        "check": "Проверить, содержит ли новость условие о форме собственности и выполняются ли остальные условия.",
        "limitations": ["ownership_does_not_establish_state_support"],
    },
    "company_age_band": {
        "meaning": "Диапазон возраста компании на неизвестную дату снимка. Возраст не устанавливает рост, зрелость, кризис или возраст группы.",
        "topics": ["regulation", "capex"],
        "check": "Для программы с ограничением по возрасту уточнить дату регистрации и дату, на которую проверяется условие.",
        "limitations": ["company_age_snapshot_unknown", "company_age_is_not_business_stage"],
    },
    "business_stage_label": {
        "meaning": "Исходная метка стадии; методика и дата неизвестны. Для изменения управленческого фокуса нужно подтвердить фактические цели и планы.",
        "topics": ["consumer_demand", "capex", "labor", "inflation"],
        "check": "Уточнить текущую цель: рост, удержание маржи или сокращение затрат; метка не доказывает финансовое состояние.",
        "limitations": ["business_stage_method_and_date_unknown", "stage_is_not_financial_health"],
    },
    "negative_mark_raw": {
        "meaning": "Исходный негативный признак с неустановленной методикой и датой. Не является подтверждением банкротства, дефолта или нарушения.",
        "topics": ["regulation"],
        "check": "До вывода о риске установить расшифровку признака и независимое подтверждение конкретного события.",
        "limitations": ["negative_mark_semantics_unknown", "no_default_or_bankruptcy_inference"],
    },
    "employees_fns_band": {
        "meaning": "Диапазон численности из поля ФНС. Период не указан; точное число сотрудников, фонд оплаты труда и текущий штат неизвестны.",
        "topics": ["labor", "regulation"],
        "check": "Проверить отчётный период и численный порог применимости. При пересечении порога диапазоном нужен точный показатель.",
        "limitations": ["employee_reporting_period_unknown", "employee_band_is_not_exact_headcount"],
    },
    "employees_spark_band": {
        "meaning": "Отдельный диапазон численности СПАРК; не заменяет данные ФНС при расхождении источников или периодов.",
        "topics": ["labor", "regulation"],
        "check": "Сверить источник и период с ФНС до проверки порога или вывода о динамике численности.",
        "limitations": ["employee_reporting_period_unknown", "employee_sources_may_disagree"],
    },
    "employees_band": {
        "meaning": "Согласованный диапазон из доступных источников. Это контекст размера работодателя; он не устанавливает трудоёмкость, зарплаты и планы найма.",
        "topics": ["labor", "regulation"],
        "check": "Сопоставить диапазон с условиями новости; для денежного эффекта запросить фонд оплаты труда, загрузку и долю трудовых затрат.",
        "limitations": ["employee_reporting_period_unknown", "headcount_is_not_payroll_or_labor_intensity"],
    },
    "city_registration": {
        "meaning": "Город регистрации из исходника. Расположение площадок, клиентов и маршрутов отдельно не подтверждено.",
        "topics": ["infrastructure", "logistics", "regulation"],
        "check": "Проверить фактическую территорию действия события и наличие операций компании на этой территории.",
        "limitations": ["registration_is_not_operational_geography"],
    },
    "client_type": {
        "meaning": "Отношения компании с банком. Этот признак не устанавливает стадию развития бизнеса, наличие денег или кредитный риск.",
        "topics": [],
        "check": "Использовать для банковского обслуживания; не повышать бизнес-релевантность новости по клиентскому статусу.",
        "limitations": ["bank_relationship_is_not_business_development_stage"],
    },
}


def factor_checks(company):
    """Evidence-linked questions, never assumed eligibility or a relevance score."""
    return [{"factor": name, "topics": policy["topics"],
             "company_feature_ids": company["factors"][name]["feature_ids"],
             "check": policy["check"], "limitations": policy["limitations"]}
            for name, policy in FACTOR_POLICIES.items() if name in company.get("factors", {}) and policy["topics"]]
