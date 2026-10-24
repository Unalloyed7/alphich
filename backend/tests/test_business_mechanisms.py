"""Business-impact analysis is open to every evidenced company consequence."""
from copy import deepcopy

import pytest

from app.intelligence.evidence import verify_draft
from app.intelligence.financial_grounding import BUSINESS_RULES, business_focus, focus_issues
from app.intelligence.schemas import CardDraft


def context(title, *, topic='sector', code='46.90', extra=()):
    result = {
        'news': {'kind': 'event_fact', 'event_id': 'current', 'text': title,
                 'attributes': {'topics': [topic]}},
    }
    if code:
        result['industry'] = {'kind': 'company_feature', 'name': 'okved_code', 'value': code,
                              'text': f'ОКВЭД: {code}'}
    result.update({key: {'kind': 'company_feature', 'name': name, 'value': value,
                        'text': f'{name}: {value}'} for key, name, value in extra})
    return result


def inference(text, company_id):
    return CardDraft.model_validate({'claims': [{'id': 'c1', 'kind': 'model_inference',
        'text': text, 'evidence_ids': ['news', company_id],
        'conditions': ['Область воздействия на компанию требует проверки по первичным данным']}]})


@pytest.mark.parametrize('title,feature_name,value,consequence', [
    ('Поставщик раскрыл утечку переписки пользователей', 'supplier_service', 'Облачная почта поставщика',
     'Если компания использует затронутую почту для обращений клиентов, раскрытие переписки может подорвать доверие; круг затронутых данных нужно проверить.'),
    ('Отозвана лицензия распространённого программного компонента', 'software_component', 'Затронутый компонент используется в продукте',
     'Если поставляемая версия подпадает под отзыв, продолжение распространения может нарушать лицензионные условия; применимость отзыва нужно проверить.'),
    ('Обнаружен способ обхода проверки подлинности сервиса', 'authentication_stack', 'Компания использует затронутый сервис',
     'Если уязвимая версия доступна извне, обход проверки может дать доступ к данным компании; версию и конфигурацию нужно проверить.'),
    ('Производитель объявил отзыв защитного оборудования', 'protective_equipment', 'Затронутая модель используется на площадке',
     'Если отозванная партия используется на площадке, защита работников может быть недостаточной; состав партии нужно проверить до продолжения работ.'),
    ('Загрязнение водозабора затронуло район производства', 'operating_region', 'Производство в затронутом районе',
     'Если производство использует затронутый водозабор, ограничение воды может нарушить работу площадки; источник водоснабжения нужно подтвердить.'),
    ('Профессиональное объединение изменило правила аттестации', 'required_certification', 'Сотрудники проходят эту аттестацию',
     'Если действующая аттестация не соответствует новым правилам, доступ сотрудников к работам может быть ограничен; переходные условия нужно проверить.'),
    ('Партнёр изменил правила доступа к закрытому каталогу', 'partner_access', 'Доступ через затронутого партнёра',
     'Если новые правила распространяются на действующий доступ, компания может потерять доступ к каталогу; применимость условий нужно проверить.'),
])
def test_nonfinancial_and_unlisted_impacts_need_no_financial_topic_or_industry(
        title, feature_name, value, consequence):
    evidence = context(title, topic='unlisted_business_subject', code=None,
                       extra=[('exposure', feature_name, value)])
    before = deepcopy(evidence)
    focus = business_focus(evidence, event_id='current', event_title=title)
    assert evidence == before
    assert focus['channel'] == 'open_business_impact'
    assert focus['news_ids'] == ['news'] and focus['company_ids'] == ['exposure']
    assert focus['industry_basis_ids'] == []
    assert focus['selection_policy']['impact_domains_open'] is True
    card = inference(consequence, 'exposure')
    assert focus_issues(card, evidence, event_id='current', event_title=title) == []
    assert verify_draft(card, evidence) == []


@pytest.mark.parametrize('title,topic', [
    ('Покупатели увеличили отсрочки оплаты поставщикам', 'sector'),
    ('Изменились сроки доставки грузов', 'logistics'),
    ('Введён инвестиционный вычет для модернизации оборудования', 'regulation'),
    ('Сталь подорожала на внутреннем рынке', 'commodities'),
    ('На рынке усилился дефицит кадров', 'labor'),
    ('Изменились обязательные требования к маркировке', 'regulation'),
])
def test_financial_and_operational_topics_remain_candidates_without_server_chosen_mechanism(title, topic):
    evidence = context(title, topic=topic)
    focus = business_focus(evidence, event_id='current', event_title=title)
    assert focus['news_ids'] == ['news'] and focus['company_ids'] == ['industry']
    assert focus['max_inferences'] == 2
    assert focus['channel'] == 'open_business_impact'
    assert focus['not_inference_basis_ids'] == []
    assert focus['decision_checks'] and focus['unknowns']


def test_employee_band_is_preserved_without_claiming_a_current_payroll_or_demanding_okved():
    title = 'На рынке усилился дефицит кадров'
    evidence = context(title, topic='labor', code=None, extra=[
        ('staff', 'employees_fns_band', '101–250'),
        ('payroll', 'CL_PRODUCT_SAL_MARK', True),
    ])
    focus = business_focus(evidence, event_title=title)
    assert focus['company_ids'] == ['staff', 'payroll']
    assert focus['company_context'][0]['company_ids'] == ['staff']
    assert 'Не текущий штат' in focus['company_context'][0]['meaning']
    card = inference('Если вакансии долго не закрываются, нехватка нужных специалистов может ограничить выполнение заказов; фактическая загрузка и вакансии неизвестны.', 'staff')
    assert focus_issues(card, evidence, event_title=title) == []


@pytest.mark.parametrize('field', ['company_age_band', 'business_stage_label', 'negative_mark_raw', 'client_type', 'group_id'])
def test_descriptive_or_opaque_mark_cannot_be_the_only_company_basis_for_a_business_effect(field):
    evidence = context('Состоялась встреча предпринимателей', extra=[('mark', field, 'A')])
    card = inference('Если компания растёт, встреча может дать ей новые заказы.', 'mark')
    assert focus_issues(card, evidence) == ['c1:business_inference_needs_operating_basis']
    assert focus_issues(card, evidence, question='Что известно о стадии компании?') == []


def test_source_semantics_are_preserved_for_registration_age_stage_and_products():
    evidence = context('Партнёр изменил условия доступа', extra=[
        ('city', 'city_registration', 'Москва'), ('age', 'company_age_band', 'до года'),
        ('stage', 'business_stage_label', 'A'), ('loan', 'CL_PRODUCT_LOA_MARK', True)])
    focus = business_focus(evidence)
    assert set(focus['company_ids']) == {'industry', 'city', 'age', 'stage', 'loan'}
    assert any('Только место регистрации' in item['meaning'] for item in focus['company_context'])
    assert any('не самостоятельное доказательство' in item['meaning'] for item in focus['company_context'])
    assert any('не действующий долг' in item['meaning'] for item in focus['company_context'])
    assert focus['not_inference_basis_ids'] == []


def test_unrelated_or_explicitly_historical_fact_cannot_establish_the_current_event():
    title = 'Раскрыта новая уязвимость программного компонента'
    evidence = context('Ранее сообщалось об уязвимости компонента', topic='sector')
    evidence['related'] = {**evidence['news'], 'text': title, 'event_id': 'other'}
    focus = business_focus(evidence, event_id='current', event_title=title)
    assert focus['news_ids'] == [] and focus['max_inferences'] == 0
    card = inference('Если используется уязвимая версия, возможен доступ к данным компании.', 'industry')
    assert 'c1:business_focus_needs_primary_news_basis' in focus_issues(card, evidence, event_id='current')


def test_mixed_event_preserves_primary_facts_from_different_impact_domains():
    evidence = context('Производитель отзывает продукцию после инцидента безопасности',
                       code=None, extra=[('product', 'product_line', 'Затронутая продукция')])
    evidence['response'] = {'kind': 'event_fact', 'event_id': 'current',
                            'text': 'Покупатели потребовали раскрыть результаты расследования.',
                            'attributes': {'topics': ['unlisted_business_subject']}}
    focus = business_focus(evidence, event_id='current')
    assert focus['news_ids'] == ['news', 'response']
    assert focus['company_ids'] == ['product']
    card = inference('Если компания выпускает затронутую продукцию, продолжение использования может угрожать безопасности покупателей; партию нужно проверить.', 'product')
    second = card.claims[0].model_copy(update={'id': 'c2', 'text':
        'Если обращения покупателей касаются продукции компании, непрозрачное расследование может подорвать доверие; состав обращений требует проверки.',
        'evidence_ids': ['response', 'product']})
    card.claims.append(second)
    assert focus_issues(card, evidence, event_id='current') == []


def test_guidance_does_not_require_financial_consequence_or_a_closed_taxonomy():
    assert 'Репутационное, правовое или техническое последствие' in BUSINESS_RULES
    assert 'Это примеры, а не закрытая классификация' in BUSINESS_RULES
    assert 'Не своди любой эффект к деньгам' in BUSINESS_RULES
