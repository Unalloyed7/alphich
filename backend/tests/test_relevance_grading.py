"""Business relevance caps: geography and materiality cannot be compensated by keywords."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from alphich_kb.personalization.grading import validate_assessment
from app.intelligence.offline import evaluate_corpus

FIXTURES=Path(__file__).resolve().parents[2]/"knowledge/fixtures"


def assessment(**changes):
    return {"industry_fit":"direct","geographic_fit":"national","materiality":"material",
        "event_status":"reported","timeliness_confirmed":True,
        "business_channel":"Изменение цен основной товарной категории на применимом рынке",**changes}


def test_only_grades_4_and_5_are_selected():
    assert validate_assessment(4,assessment())
    assert not validate_assessment(3,assessment(materiality="unknown"))
    assert validate_assessment(5,assessment(event_status="in_force",applicability_confirmed=True,
        required_action="Проверить обязательное оформление перевозочных документов."))


@pytest.mark.parametrize("changes",[
    {"geographic_fit":"other_region"},{"geographic_fit":"unknown"},{"geographic_fit":"registration_only"},
    {"geographic_fit":"not_applicable"},{"materiality":"small"},{"materiality":"unknown"},
    {"industry_fit":"adjacent"},{"industry_fit":"none"},{"event_status":"proposal"},
    {"event_status":"announcement"},{"timeliness_confirmed":False}])
def test_grade_4_requires_each_geographic_and_materiality_gate(changes):
    with pytest.raises(ValueError):
        validate_assessment(4,assessment(**changes))


def test_grade_5_requires_confirmed_direct_action_not_general_industry_news():
    with pytest.raises(ValueError):
        validate_assessment(5,assessment())
    assert validate_assessment(5,assessment(direct_company_event_confirmed=True,
        required_action="Проверить исполнение подтверждённого договора компании."))


@pytest.mark.parametrize("bad",[-1,6,True,4.0,"5"])
def test_grade_type_and_scale_are_strict(bad):
    with pytest.raises(ValueError): validate_assessment(bad,assessment())


def test_new_50_news_and_150_labels_have_no_overlap_with_previous_set():
    new=json.loads((FIXTURES/"real_news_selection_labels.v2.json").read_text())
    old=json.loads((FIXTURES/"real_news_selection_labels.v1.json").read_text())
    assert len(new["sources"])==50 and len(new["labels"])==150
    assert not set(new["company_ids"]) & set(old["company_ids"])
    assert not {s["event_id"] for s in new["sources"]} & {s["event_id"] for s in old["sources"]}
    assert not {s["url"] for s in new["sources"]} & {s["url"] for s in old["sources"]}
    for row in new["labels"]:
        assert validate_assessment(row["grade"],row["assessment"]) == row["selected"]
    assert new["grade_scale"]=={"min":0,"max":5} and new["selection_threshold"]==4


def test_evaluator_uses_4_as_relevance_cutoff_and_preserves_legacy_scale(tmp_path):
    profiles=FIXTURES/"company_profiles.json"
    event={"id":"one","version":1,"company_id":None,"title":"Банк России сохранил ключевую ставку.",
        "available_at":"2026-10-20T10:00:00Z","sources":[{"id":"src","version":1,
        "text":"Банк России сохранил ключевую ставку. Условия договоров неизвестны.",
        "available_at":"2026-10-20T10:00:00Z"}]}
    events=tmp_path/"events.json"; events.write_text(json.dumps({"events":[event]}))
    labels=tmp_path/"labels.json"
    data={"company_ids":["S_BUILD"],"roles":["cfo"],"as_of":"2026-10-21T12:00:00Z",
        "grade_scale":{"min":0,"max":5},"selection_threshold":4,
        "labels":[{"company_id":"S_BUILD","event_id":"one","grade":3,
            "assessment":assessment(materiality="unknown"),"selected":False}]}
    labels.write_text(json.dumps(data))
    report=evaluate_corpus(profiles,events,labels)
    result=report["modes"]["lexical"]["results"][0]
    assert report["selection_threshold"]==4 and result["precision_at_5"]==0
    assert result["below_selection_threshold"]==len(result["items"])
    data["selection_threshold"]=2; labels.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="only_4_and_5"): evaluate_corpus(profiles,events,labels)


def test_positive_extension_has_new_sources_and_explicit_difficult_negatives():
    extension=json.loads((FIXTURES/"real_news_selection_labels.v3.json").read_text())
    previous=json.loads((FIXTURES/"real_news_selection_labels.v2.json").read_text())
    assert len(extension["sources"])==12 and len(extension["labels"])==36
    assert extension["positive_pairs"]==6
    assert not {s["url"] for s in extension["sources"]}&{s["url"] for s in previous["sources"]}
    for row in extension["labels"]:
        assert validate_assessment(row["grade"],row["assessment"])==row["selected"]
