"""Conservative local reprint grouping, after authorization and time filtering."""
import re
from .inputs import instant, words
from ..common import digest

REPRINT_VERSION = "reprints-v1-shingles-numeric-guard"


def _signature(event):
    body = words(event["text"])
    tokens = body.split()
    shingles = {tuple(tokens[i:i+5]) for i in range(max(0, len(tokens)-4))}
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", event["title"] + " " + event["text"]))
    directions = {key for key, pattern in {"up": r"повыс|вырос|увелич", "down": r"сниз|сократ|упал",
        "hold": r"сохрани|неизмен", "negation": r"\bне\b|\bотмен|\bзапрет|\bприостанов"}.items() if re.search(pattern, body)}
    dimensions = {digest({k:f["attributes"].get(k) for k in ("kind","topics","geography","forecast_horizon")}) for f in event["facts"]}
    entities = set(re.findall(r'«([^»]+)»|"([^"\n]+)"', event["title"]+" "+event["text"]))
    return body, shingles, numbers, directions, dimensions, entities


def group_reprints(events):
    groups, excluded = [], {"exact_reprint": 0, "near_reprint": 0}
    signatures = {e["id"]: _signature(e) for e in events}
    for event in sorted(events, key=lambda e: (instant(e["available_at"]), e["id"])):
        body, shingles, numbers, directions, dimensions, entities = signatures[event["id"]]
        target = None
        for group in groups:
            anchor = group[0]
            other, other_shingles, other_numbers, other_directions, other_dimensions, other_entities = signatures[anchor["id"]]
            if numbers != other_numbers or directions != other_directions or entities != other_entities:
                continue
            if dimensions and other_dimensions and dimensions != other_dimensions:
                continue
            if abs((instant(event["published_at"] or event["available_at"]) -
                    instant(anchor["published_at"] or anchor["available_at"])).total_seconds()) > 48*3600:
                continue
            exact = body == other
            union = shingles | other_shingles
            similar = len(shingles) >= 25 and len(other_shingles) >= 25 and bool(union) and len(shingles & other_shingles)/len(union) >= .88
            if exact or similar:
                target = group
                excluded["exact_reprint" if exact else "near_reprint"] += 1
                break
        if target is None:
            groups.append([event])
        else:
            target.append(event)
    result = []
    for group in groups:
        representative = min(group, key=lambda e: (-len(e["facts"]), instant(e["available_at"]), e["id"]))
        representative["duplicate_event_ids"] = sorted(e["id"] for e in group if e["id"] != representative["id"])
        representative["story_group_id"] = "story_" + digest([REPRINT_VERSION, group[0]["snapshot_hash"]])
        representative["reprint_sources"] = [{"event_id": e["id"], "source_id": s["id"], "title": e["title"],
            "url": s["url"], "published_at": s["published_at"]} for e in sorted(group, key=lambda e:e["id"])
            for s in e["sources"]][:20]
        representative["reprint_group_size"] = len(group)
        representative["age_anchor"] = min(e["age_anchor"] for e in group)
        result.append(representative)
    return result, excluded
