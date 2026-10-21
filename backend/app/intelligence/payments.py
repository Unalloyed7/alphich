"""Optional read-only bridge to an ETL snapshot; DB import can use attach_payment_context too."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path


@lru_cache(maxsize=2)
def _read(directory, manifest_stamp, contexts_stamp):
    from alphich_kb.common import KnowledgeError
    path = Path(directory)
    if (path/"contexts.jsonl").stat().st_size > 32*1024*1024:
        raise KnowledgeError("Payment snapshot too large")
    manifest = json.loads((path/"manifest.json").read_text())
    raw = (path/"contexts.jsonl").read_bytes()
    if len(raw) > 32*1024*1024 or hashlib.sha256(raw).hexdigest() != manifest["files"]["contexts.jsonl"]:
        raise KnowledgeError("Payment snapshot checksum mismatch")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) > 10000 or len({r["company_id"] for r in rows}) != len(rows):
        raise KnowledgeError("Invalid payment snapshot identities")
    for row in rows:
        if row["context_version"] != manifest["context_version"] or row["profile_version"] != manifest["dataset_version"]:
            raise KnowledgeError("Mixed payment snapshot versions")
        inputs={item["name"]:item["sha256"] for item in manifest["inputs"]}
        for feature in row["features"]:
            if feature["name"] in {"payment_incoming","payment_outgoing"} and any(
                    ref.get("sha256")!=inputs.get(ref.get("dataset")) for ref in feature["source_refs"]):
                raise KnowledgeError("Payment provenance disagrees with manifest")
    return {r["company_id"]: r for r in rows}


def enrich_payments(profile, context=None):
    from alphich_kb.common import KnowledgeError
    from alphich_kb.personalization.payments import attach_payment_context
    directory = os.getenv("COMPANY_CONTEXT_DIR", "")
    if context is None and not directory:
        return profile
    try:
        if context is None:
            path = Path(directory)
            context = _read(str(path.resolve()), (path/"manifest.json").stat().st_mtime_ns,
                            (path/"contexts.jsonl").stat().st_mtime_ns).get(profile["id"])
        if context is None:
            raise KnowledgeError("Company missing from payment snapshot")
        return attach_payment_context(profile, context)
    except (KnowledgeError, OSError, ValueError, KeyError, TypeError):
        return {**profile, "limitations": [*profile.get("limitations", []), "payment_context_unavailable_or_mismatched"]}
