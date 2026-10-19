"""Build token-bounded search chunks and manually verified claim evidence."""

import re
from datetime import date
from pathlib import Path

from .common import (KnowledgeError, artifact_dir, digest, file_hash, read_json,
                     read_jsonl, seal, verify_artifacts, write_json, write_jsonl)

CHUNKER_VERSION = "e5-page-spans-v1"


class E5Tokenizer:
    def __init__(self, tokenizer_path, expected_hash=None):
        from tokenizers import Tokenizer, __version__
        actual_hash = file_hash(tokenizer_path)
        if expected_hash and actual_hash != expected_hash:
            raise KnowledgeError("Tokenizer checksum mismatch")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Never inherit silent truncation/padding from a downloaded config.
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.metadata = {"model_id": "intfloat/multilingual-e5-base",
                         "tokenizer_sha256": actual_hash, "library": f"tokenizers=={__version__}"}

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True).ids)


def chunk_spans(text, header, counter, max_tokens=384):
    """Exact page offsets, semantic boundaries, bounded overlap, no lost text."""
    if not 64 <= max_tokens <= 512:
        raise KnowledgeError("Chunk token budget must be 64..512")
    if counter.count(header) >= max_tokens - 8:
        raise KnowledgeError("Document metadata leaves no room for a chunk")
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            break
        low, high = start + 1, len(text)
        if counter.count(header + text[start:low]) > max_tokens:
            raise KnowledgeError("Cannot fit a single character in the token budget")
        while low < high:
            mid = (low + high + 1) // 2
            if counter.count(header + text[start:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        end = low
        if end < len(text):
            # Prefer whole paragraphs, sentences, lines, then words.
            minimum = start + int((end - start) * 0.6)
            for pattern in (r"\n\n", r"(?<=[.!?])\s+", r"\n", r"\s+"):
                candidates = list(re.finditer(pattern, text[minimum:end]))
                if candidates:
                    end = minimum + candidates[-1].start()
                    break
        while end > start and text[end - 1].isspace():
            end -= 1
        # Token counts are checked again after choosing a semantic boundary.
        while end > start and counter.count(header + text[start:end]) > max_tokens:
            end -= 1
        if end <= start:
            raise KnowledgeError("Chunker made no progress")
        yield start, end
        if end == len(text):
            break
        # Repeat at most the final sentence/line, if it fits in a quarter chunk.
        boundaries = list(re.finditer(r"(?<=[.!?])\s+|\n", text[start:end]))
        overlap_start = start + boundaries[-1].end() if boundaries else end
        if overlap_start <= start or counter.count(text[overlap_start:end]) > max_tokens // 4:
            overlap_start = end
        start = overlap_start


def page_header(doc, page):
    return f"passage: {doc['title']} | {doc['published_at'] or 'publication date unknown'} | p.{page}\n"


def validate_date(doc):
    if doc["published_at"] is not None:
        date.fromisoformat(doc["published_at"])
        if doc["publication_precision"] != "day":
            raise KnowledgeError("A publication day needs day precision")
    elif doc["publication_precision"] != "unknown":
        raise KnowledgeError("Unknown date cannot be inferred from the period label")


def make_chunk(doc, page, start, end, config, counter, reviewed=False, section="content"):
    text = page["text"][start:end]
    text_version = digest({"page_text_version": page["text_version"], "start": start, "end": end})
    identity = digest({"text_version": text_version, "config": config, "reviewed": reviewed, "section": section})
    embedding_text = page_header(doc, page["page"]) + text
    lines = [line for line in page["lines"] if line["start"] < end and line["end"] > start]
    return {
        "id": f"chunk_{identity}", "kind": "document_chunk", "document_id": doc["id"],
        "document_version": doc["version"], "source_version": doc["version"], "parent_id": page["id"],
        "page": page["page"], "source_span": {"start": start, "end": end, "unit": "unicode_codepoints"},
        "line_bboxes": [{"start": line["start"], "end": line["end"], "bbox": line["bbox"]} for line in lines],
        "title": doc["title"], "url": doc["url"], "published_at": doc["published_at"],
        "publication_precision": doc["publication_precision"], "content_hash": doc["content_hash"],
        "text": text, "text_version": text_version, "section": section,
        "topics": doc["topics"], "topics_scope": "document_inherited", "forecast_horizon": None,
        "ocr_checked": reviewed, "extraction_status": "ocr_span_reviewed" if reviewed else "ocr_unreviewed",
        "quality_flags": [] if reviewed else page["quality_flags"],
        "embedding_text": embedding_text, "embedding_token_count": counter.count(embedding_text),
        "source_refs": [{"document_id": doc["id"], "url": doc["url"], "page": page["page"],
                         "content_hash": doc["content_hash"], "page_text_version": page["text_version"],
                         "start": start, "end": end, "image": page["image"]}],
    }


def prepare_claims(annotations, documents, pages, config, counter):
    """A review is valid only for the exact PDF, OCR text and quoted substring."""
    evidence, claims, logical_ids = [], [], set()
    by_doc = {d["id"]: d for d in documents}
    by_page = {(p["document_id"], p["page"]): p for p in pages}
    for review in annotations.get("claims", []):
        doc = by_doc[review["document_id"]]
        page = by_page[(doc["id"], review["page"])]
        if review["id"] in logical_ids:
            raise KnowledgeError("Duplicate claim annotation ID")
        logical_ids.add(review["id"])
        if review["pdf_sha256"] != doc["content_hash"] or review["page_text_version"] != page["text_version"]:
            raise KnowledgeError(f"Stale visual review: {review['id']}")
        quote = review["quote"]
        if not quote or page["text"].count(quote) != 1:
            raise KnowledgeError(f"Quote must occur exactly once: {review['id']}")
        if review["verification_status"] != "visually_verified" or not review.get("review_note"):
            raise KnowledgeError("Claims require an explicit visual review note")
        if review["kind"] not in {"observation", "interpretation", "forecast", "mechanism"}:
            raise KnowledgeError("Invalid bank claim kind")
        if review["position_role"] not in {"current_in_document", "previous_view", "conditional"}:
            raise KnowledgeError("Invalid bank position role")
        if review["kind"] == "forecast" and not review.get("forecast_horizon"):
            raise KnowledgeError("Forecast needs a horizon")
        if not review.get("attributed_to") or not review.get("metric"):
            raise KnowledgeError("Claim author and metric are required")
        start = page["text"].index(quote)
        chunk = make_chunk(doc, page, start, start + len(quote), config, counter, reviewed=True, section="reviewed_evidence")
        if chunk["embedding_token_count"] > 512:
            raise KnowledgeError("Reviewed quote exceeds E5 input; select a smaller exact span")
        evidence.append(chunk)
        fields = {key: review.get(key) for key in (
            "id", "kind", "attributed_to", "metric", "geography", "scenario", "value", "unit",
            "observed_period", "forecast_horizon", "position_role", "revision_of", "verification_status", "review_note")}
        claims.append({**fields, "annotation_id": review["id"], "id": f"claim_{digest({'review': review, 'evidence': chunk['id']})}",
                       "document_id": doc["id"], "document_chunk_id": chunk["id"], "published_at": doc["published_at"],
                       "text": review["statement"], "quote": quote})
    ids = {c["annotation_id"]: c for c in claims}
    for claim in claims:
        previous_id = claim["revision_of"]
        if previous_id:
            if previous_id not in ids:
                raise KnowledgeError(f"Unknown revision target: {previous_id}")
            previous = ids[previous_id]
            if claim["kind"] != "forecast" or previous["kind"] != "forecast" or previous["position_role"] != "current_in_document":
                raise KnowledgeError("A revision must point to a forecast current in its original document")
            if not previous["published_at"] or not claim["published_at"] or previous["published_at"] >= claim["published_at"]:
                raise KnowledgeError("Revision must follow an earlier dated document")
            for key in ("metric", "geography", "scenario", "forecast_horizon", "attributed_to", "unit"):
                if previous[key] != claim[key]:
                    raise KnowledgeError(f"Revision changes {key}")
            claim["revision_of"] = previous["id"]
    return list({item["id"]: item for item in evidence}.values()), claims


def build(ocr_dir, output_dir, tokenizer_path, annotations_path=None, max_tokens=384, tokenizer_hash=None):
    ocr_manifest = verify_artifacts(ocr_dir)
    counter = E5Tokenizer(tokenizer_path, tokenizer_hash)
    documents = read_jsonl(Path(ocr_dir) / "documents.jsonl")
    pages = read_jsonl(Path(ocr_dir) / "pages.jsonl")
    annotations = read_json(annotations_path) if annotations_path else {"claims": [], "page_sections": []}
    sections = {}
    for entry in annotations.get("page_sections", []):
        page = next((p for p in pages if (p["document_id"], p["page"]) == (entry["document_id"], entry["page"])), None)
        if page is None or entry["page_text_version"] != page["text_version"]:
            raise KnowledgeError("Stale page classification")
        if entry["section"] not in {"content", "disclaimer", "contacts", "cover"}:
            raise KnowledgeError("Invalid page section")
        sections[(entry["document_id"], entry["page"])] = entry["section"]
    config = {"version": CHUNKER_VERSION, "max_tokens": max_tokens, "tokenizer": counter.metadata,
              "overlap": "last_sentence_or_line_up_to_quarter_budget", "prefix": "passage: ",
              "revision": "d128750597153bb5987e10b1c3493a34e5a4502a"}
    chunks, by_doc = [], {d["id"]: d for d in documents}
    for doc in documents:
        validate_date(doc)
    for page in pages:
        doc = by_doc[page["document_id"]]
        section = sections.get((doc["id"], page["page"]), "content")
        if section in {"disclaimer", "contacts", "cover"}:
            continue
        for start, end in chunk_spans(page["text"], page_header(doc, page["page"]), counter, max_tokens):
            chunks.append(make_chunk(doc, page, start, end, config, counter, section=section))
    evidence, claims = prepare_claims(annotations, documents, pages, config, counter)
    version = digest({"ocr_version": ocr_manifest["version"], "config": config, "annotations": annotations})
    quality = {"documents": len(documents), "pages": len(pages), "search_chunks": len(chunks),
               "reviewed_evidence_spans": len(evidence), "verified_claims": len(claims),
               "undated_documents": [d["id"] for d in documents if not d["published_at"]],
               "excluded_pages": len([v for v in sections.values() if v != "content"]),
               "low_confidence_pages": [{"document_id": p["document_id"], "page": p["page"]} for p in pages if "low_ocr_confidence" in p["quality_flags"]],
               "max_embedding_tokens": max((c["embedding_token_count"] for c in chunks + evidence), default=0)}
    with artifact_dir(output_dir) as stage:
        write_jsonl(stage / "documents.jsonl", documents)
        write_jsonl(stage / "pages.jsonl", pages)
        write_jsonl(stage / "chunks.jsonl", chunks)
        write_jsonl(stage / "evidence.jsonl", evidence)
        write_jsonl(stage / "claims.jsonl", claims)
        write_json(stage / "quality.json", quality)
        seal(stage, {"kind": "bank_knowledge", "schema_version": 1, "version": version, "config": config,
                     "ocr_version": ocr_manifest["version"], "ocr_artifacts": str(ocr_dir),
                     "annotations_hash": file_hash(annotations_path) if annotations_path else None})
    return {**quality, "output_dir": str(output_dir), "version": version}
