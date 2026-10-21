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
Если обоснованного канала нет, оставь факты и missing_data; не сочиняй персональный эффект.
"""


def business_focus(registry, *, question=None, event_id=None):
    """A construction/rate hypothesis to test, never a fact about customer exposure.

    Other sectors keep the general business reasoning rules. Direct questions
    retain their requested subject instead of being diverted to this lens.
    """
    if question:
        return None
    news = [key for key, e in registry.items() if e.get("kind") == "event_fact"
            and (event_id is None or e.get("event_id") == event_id)
            and "policy_rate" in e.get("attributes", {}).get("topics", [])]
    industry = [key for key, e in registry.items() if e.get("kind") == "company_feature"
                and e.get("name") == "okved_code" and str(e.get("value", "")).split(".")[0] in {"41", "42", "43"}]
    if not news or not industry:
        return None
    industry += [key for key, e in registry.items() if e.get("kind") == "company_feature"
                 and e.get("name") == "okved_name"]
    return {"channel": "customer_investment_demand", "news_ids": news, "company_ids": industry,
            "hypothesis": "Если запуск проектов заказчиков зависит от кредита, длительная высокая ставка может откладывать запуск и поступления строительной компании.",
            "unknowns": ["Зависимость проектов заказчиков от кредитов", "Состав заказчиков и сроки запуска", "Условия и сроки оплаты"],
            "decision": "Надёжность плана заказов и поступлений; проверить финансирование и сроки запуска крупных заявок.",
            "constraints": "Разбери этот канал, явно условно. Наличие долга самой компании и её проектного финансирования не установлено. Не заменяй канал спроса рассуждением о её кредите.",
            "not_inference_basis_ids": [key for key, e in registry.items() if e.get("name") == "CL_PRODUCT_LOA_MARK"]}


def focus_issues(draft, registry, *, question=None, event_id=None):
    focus = business_focus(registry, question=question, event_id=event_id)
    if not focus:
        return []
    issues = []
    for claim in draft.claims:
        if claim.kind != "model_inference":
            continue
        if not set(claim.evidence_ids).intersection(focus["company_ids"]):
            issues.append(f"{claim.id}:business_focus_needs_industry_basis")
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
