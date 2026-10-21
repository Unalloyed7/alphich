"""Local adapters for company/news JSON; production authorization stays in scope."""

from pathlib import Path

from ..common import KnowledgeError, read_json, read_jsonl
from .inputs import SelectionScope
from .selector import NewsSelector


def load_rows(path, field):
    if Path(path).suffix == ".jsonl":
        return read_jsonl(path)
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if field in payload:
        return payload[field]
    if "id" in payload or "company_id" in payload:
        return [payload]
    raise KnowledgeError(f"Expected a JSON list or {field!r} field")


def load_encoder(args):
    if args.mode == "lexical":
        return None
    try:
        from ..embeddings import E5Encoder, EncoderUnavailable
    except ImportError:
        return None
    try:
        return E5Encoder(args.model_dir, args.model_manifest, args.threads)
    except EncoderUnavailable:
        return None


def select_from_files(args):
    profiles = load_rows(args.profiles, "companies")
    selected = [p for p in profiles if p.get("company_id", p.get("id")) == args.company_id]
    if len(selected) != 1:
        raise KnowledgeError("Select exactly one company profile with --company-id")
    events = load_rows(args.events, "events")
    encoder = load_encoder(args)
    selector = NewsSelector(events, encoder)
    allowed = args.allow_event if args.allow_event is not None else [e["id"] for e in events]
    scope = SelectionScope(args.company_id, args.as_of, frozenset(allowed), args.role)
    result = selector.select(selected[0], scope, mode=args.mode, limit=args.limit, min_grade=args.min_grade)
    if args.bank_index_dir:
        from ..context import context_for_model
        from ..retriever import BankRetriever, RetrievalScope
        bank = BankRetriever(args.bank_index_dir, encoder)
        allowed_bank = args.allow_bank_document if args.allow_bank_document is not None else bank.documents.keys()
        ready = [item for item in result["items"] if item["status"] == "applicable_with_conditions"]
        if args.bank_event_id:
            ready = [item for item in ready if item["event_id"] == args.bank_event_id]
            if not ready:
                raise KnowledgeError("The requested bank event is not a selected applicable event")
        contexts = []
        if ready:
            for question in ready[0]["bank_questions"]:
                response = bank.retrieve(question["query"], RetrievalScope(question["as_of"], frozenset(allowed_bank)),
                                         mode="lexical" if args.mode == "lexical" else "dense",
                                         metric=question["metric"], horizon=question["horizon"], max_context_chars=6000)
                contexts.append({"question_id": question["id"], "event_id": ready[0]["event_id"],
                                 "context": context_for_model(response)})
        result["bank_contexts"] = contexts
        result["bank_stage"] = "context_prepared" if contexts else "awaiting_event_extraction"
    return result
