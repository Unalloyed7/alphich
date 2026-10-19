"""Poppler rendering and Tesseract TSV extraction with page coordinates."""

import csv
import io
import math
import os
import re
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import (KnowledgeError, artifact_dir, digest, file_hash, read_json,
                     safe_child, seal, write_jsonl)

EXTRACTOR_VERSION = "tesseract-tsv-v1"


def run(args, timeout=180):
    env = {**os.environ, "LC_ALL": "C", "OMP_THREAD_LIMIT": "1"}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise KnowledgeError(f"Cannot run {args[0]}: {error}") from error
    if result.returncode:
        raise KnowledgeError(f"{args[0]} failed: {result.stderr[-1500:]}")
    return result.stdout or result.stderr


def parse_tsv(tsv):
    """Keep OCR reading order and explicit paragraph breaks; never repair numbers."""
    rows = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    required = {"level", "block_num", "par_num", "line_num", "left", "top", "width", "height", "conf", "text"}
    if not required.issubset(rows.fieldnames or []):
        raise KnowledgeError("Invalid Tesseract TSV header")
    lines = OrderedDict()
    width = height = 0
    for row in rows:
        if row["level"] == "1":
            width, height = int(row["width"]), int(row["height"])
        if row["level"] != "5" or not row["text"].strip():
            continue
        key = tuple(int(row[k]) for k in ("block_num", "par_num", "line_num"))
        confidence = float(row["conf"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 100:
            raise KnowledgeError("Invalid OCR word confidence")
        lines.setdefault(key, []).append({
            "text": row["text"].strip(), "confidence": confidence,
            "bbox": [int(row[k]) for k in ("left", "top", "width", "height")],
        })
    text, spans, confidences = "", [], []
    previous = None
    for key, words in lines.items():
        if text:
            text += "\n" if previous == key[:2] else "\n\n"
        start = len(text)
        text += " ".join(word["text"] for word in words)
        left = min(w["bbox"][0] for w in words)
        top = min(w["bbox"][1] for w in words)
        right = max(w["bbox"][0] + w["bbox"][2] for w in words)
        bottom = max(w["bbox"][1] + w["bbox"][3] for w in words)
        spans.append({"start": start, "end": len(text), "block": key[0], "paragraph": key[1],
                      "line": key[2], "bbox": [left, top, right - left, bottom - top], "words": words})
        confidences.extend(w["confidence"] for w in words)
        previous = key[:2]
    return {
        "text": text, "lines": spans, "width": width, "height": height,
        "word_count": len(confidences),
        "mean_word_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "low_confidence_words": sum(c < 60 for c in confidences),
    }


def validate_corpus(path):
    corpus = read_json(path)
    ids, files = set(), set()
    for doc in corpus["documents"]:
        safe_child(Path("."), doc["file"])
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", doc["id"]):
            raise KnowledgeError(f"Invalid document ID: {doc['id']}")
        if doc["id"] in ids or doc["file"] in files:
            raise KnowledgeError("Duplicate document ID or filename")
        ids.add(doc["id"])
        files.add(doc["file"])
        if not re.fullmatch(r"[0-9a-f]{64}", doc["sha256"]):
            raise KnowledgeError("Missing SHA-256")
        if not isinstance(doc["pages"], int) or not 1 <= doc["pages"] <= 500:
            raise KnowledgeError("Invalid page count")
    return corpus


def extract(corpus_path, input_dir, output_dir, tessdata_dir, dpi=240, workers=2):
    if not 100 <= dpi <= 400 or not 1 <= workers <= 4:
        raise KnowledgeError("DPI must be 100..400; workers must be 1..4")
    corpus = validate_corpus(corpus_path)
    languages = {lang: file_hash(Path(tessdata_dir) / f"{lang}.traineddata") for lang in ("rus", "eng")}
    engine = {
        "version": EXTRACTOR_VERSION, "dpi": dpi, "psm": 3, "oem": 1, "languages": languages,
        "tesseract": run(["tesseract", "--version"]).splitlines()[0],
        "poppler": run(["pdftoppm", "-v"]).splitlines()[0],
    }
    documents = []
    for source in corpus["documents"]:
        pdf = safe_child(input_dir, source["file"])
        if file_hash(pdf) != source["sha256"]:
            raise KnowledgeError(f"Source checksum mismatch: {pdf}")
        info = run(["pdfinfo", str(pdf)])
        match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
        if not match or int(match[1]) != source["pages"]:
            raise KnowledgeError(f"Page count mismatch: {pdf}")
        version = digest({"source": source, "engine": engine})
        documents.append({**source, "kind": "bank_pdf", "content_hash": source["sha256"],
                          "version": version, "extraction_status": "ocr_unreviewed"})
    with artifact_dir(output_dir) as stage:
        def process(task):
            doc, page = task
            folder = stage / "pages" / doc["id"]
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder / f"p{page:03d}"
            run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-singlefile",
                 "-png", str(safe_child(input_dir, doc["file"])), str(stem)])
            tsv = run(["tesseract", str(stem.with_suffix(".png")), "stdout", "--tessdata-dir",
                       str(tessdata_dir), "-l", "rus+eng", "--oem", "1", "--psm", "3",
                       "-c", "tessedit_create_tsv=1"])
            stem.with_suffix(".tsv").write_text(tsv, encoding="utf-8")
            parsed = parse_tsv(tsv)
            text_version = digest({"document_version": doc["version"], "page": page, "text": parsed["text"]})
            flags = ["ocr_unreviewed", "layout_requires_review"]
            if parsed["word_count"] < 10:
                flags.append("sparse_page")
            if (parsed["mean_word_confidence"] or 0) < 75:
                flags.append("low_ocr_confidence")
            result = {
                "id": f"page_{text_version}", "document_id": doc["id"], "document_version": doc["version"],
                "page": page, "text_version": text_version, **parsed, "ocr_checked": False,
                "quality_flags": flags, "image": str(stem.with_suffix(".png").relative_to(stage)),
                "tsv": str(stem.with_suffix(".tsv").relative_to(stage)),
            }
            print(f"OCR {doc['id']} {page}/{doc['pages']}: {parsed['word_count']} words", file=sys.stderr, flush=True)
            return result
        tasks = [(doc, page) for doc in documents for page in range(1, doc["pages"] + 1)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pages = list(pool.map(process, tasks))
        write_jsonl(stage / "documents.jsonl", documents)
        write_jsonl(stage / "pages.jsonl", pages)
        seal(stage, {"kind": "bank_ocr", "schema_version": 1, "engine": engine,
                     "corpus_hash": file_hash(corpus_path), "documents": len(documents), "pages": len(pages),
                     "version": digest([p["text_version"] for p in pages])})
    return {"documents": len(documents), "pages": len(pages), "output_dir": str(output_dir)}
