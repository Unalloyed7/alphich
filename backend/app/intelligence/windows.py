"""Deterministic source-coordinate excerpts under one shared prompt budget."""
import re
from bisect import bisect_left, bisect_right

WINDOW_VERSION = "source-windows-v1"
SIGNALS = re.compile(r"ставк|инфляц|пошлин|налог|экспорт|импорт|производств|выручк|инвестиц|"
                     r"цен[аыуе]|платеж|платёж|кредит|сниз|повыс|вырос|сократ|увелич|запрет|"
                     r"вступ\w* в силу|\d+(?:[.,]\d+)?\s*%", re.I)


def source_windows(text, title="", maximum_bytes=6500):
    if maximum_bytes < 512:
        raise ValueError("Source window budget must be at least 512 bytes")
    if len(text.encode()) <= maximum_bytes:
        return text, {"start": 0, "end": len(text), "total_characters": len(text),
                      "complete": True, "windows": [{"start": 0, "end": len(text)}],
                      "selected_bytes": len(text.encode()), "method": WINDOW_VERSION}
    size = (maximum_bytes - 200) // 4
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1]+len(char.encode()))
    candidates, start = [], 0
    while start < len(text):
        end = min(len(text), bisect_right(offsets,offsets[start]+size)-1)
        candidates.append((start,end))
        if end == len(text): break
        start += max(1,(end-start)//2)
    candidates = sorted(set(candidates + [(bisect_left(offsets,offsets[-1]-size),len(text))]))
    title_words = set(re.findall(r"[\w]{5,}", title.lower()))
    def score(span):
        passage = text[slice(*span)]
        return (len(SIGNALS.findall(passage)) + len(title_words & set(re.findall(r"\w+", passage.lower()))), -span[0])
    chosen = [candidates[0], candidates[-1]]
    for span in sorted(candidates[1:-1], key=score, reverse=True):
        if len(chosen) >= 4:
            break
        chosen.append(span)
    merged = []
    for start, end in sorted(chosen):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    view = "\n\n[ПРОПУСК ИСХОДНОГО ТЕКСТА]\n\n".join(text[s:e] for s, e in merged)
    assert len(view.encode()) <= maximum_bytes
    return view, {"start": merged[0][0], "end": merged[-1][1], "total_characters": len(text),
                  "complete": sum(e-s for s,e in merged) == len(text),
                  "windows": [{"start": s, "end": e} for s,e in merged],
                  "selected_bytes": len(view.encode()), "method": WINDOW_VERSION}
