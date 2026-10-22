"""Local Qwen preannotations; no gateway, paid provider, keys or database.

Only an explicitly loopback OpenAI-compatible server is accepted. Predictions
are weak teacher labels, not independently reviewed training truth. Immutable
per-pair request markers prevent automatic retries after ambiguous failures.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ..common import KnowledgeError, canonical, digest
from .dataset import (_strict_json, input_hash, iter_dataset, validate_pair,
                      write_dataset, write_private_json)
from .label_studio import qwen_prompt
from .teacher import prepare_judgement, validate_judgement

VERSION = "local-qwen-prelabel-v2-semantic-aliases"
MAX_RESPONSE_BYTES = 1024 * 1024


def local_endpoint(base_url):
    """Never use environment proxies, external DNS names or URL credentials."""
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except (ValueError, TypeError):
        raise KnowledgeError("Invalid local model URL") from None
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or "?" in base_url or "#" in base_url or parsed.path.rstrip("/") not in {"", "/v1"}
            or port is None or not 1 <= port <= 65535):
        raise KnowledgeError("Use http://127.0.0.1:PORT/v1 (or localhost/[::1]); external APIs are forbidden")
    # Resolve the literal localhost alias ourselves, rather than consulting DNS.
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return urlunsplit(("http", f"{host}:{port}", "/v1/chat/completions", "", ""))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise KnowledgeError("Local model redirects are forbidden")


def _request(endpoint, payload, timeout):
    encoded = canonical(payload).encode("utf-8")
    if len(encoded) > 2 * 1024 * 1024:
        raise KnowledgeError("Local request exceeds 2 MiB")
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    request = Request(endpoint, data=encoded, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        # Do not print an untrusted response body, headers, prompts or secrets.
        raise KnowledgeError(f"Local model returned HTTP {error.code}; no automatic retry") from None
    except (URLError, TimeoutError, OSError):
        raise KnowledgeError("Local request failed or timed out; no automatic retry") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise KnowledgeError("Local response exceeds 1 MiB")
    value = _strict_json(raw)
    try:
        choice = value["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise KnowledgeError("Local completion was truncated or did not finish normally")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise KnowledgeError("Local completion must contain a JSON text response")
    except (KeyError, TypeError, IndexError):
        raise KnowledgeError("Invalid local OpenAI-compatible response") from None
    return _strict_json(content)


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise KnowledgeError("Missing, linked or oversized local checkpoint")
    return _strict_json(path.read_bytes())


def _validated_checkpoint(path, source, manifest):
    row = validate_pair(_read(path))
    label = row.get("label", {})
    plan = prepare_judgement(source)
    if (input_hash(row) != input_hash(source) or label.get("origin") != "teacher"
            or label.get("labeler") != "local:" + manifest["model"]
            or label.get("label_version") != plan["prompt_version"]
            or label.get("prompt_hash") != manifest["prompt_hash"]
            or label.get("request_hash") != digest(qwen_prompt(source)[0])
            or label.get("batch_id") != manifest["batch_id"]
            or label.get("target") != "business_relevance"
            or label.get("independently_reviewed") is not False):
        raise KnowledgeError("Local checkpoint does not match its input/model/prompt")
    judgement = validate_judgement(label.get("judgement"), source)
    if judgement["grade"] != label.get("grade") or judgement["reason"] != label.get("reason"):
        raise KnowledgeError("Local checkpoint label conflicts with its judgement")
    return row


def label_batch(input_path, output_dir, *, model, base_url, limit=10, offset=0, timeout=120):
    if type(limit) is not int or not 1 <= limit <= 1000 or type(offset) is not int or offset < 0:
        raise KnowledgeError("Use limit 1..1000 and a nonnegative offset")
    if not isinstance(model, str) or not model.strip() or len(model) > 300:
        raise KnowledgeError("An explicit local model ID is required")
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise KnowledgeError("Request timeout must be 1..600 seconds")
    endpoint = local_endpoint(base_url)
    rows = [row for index, row in enumerate(iter_dataset(input_path, limit=offset + limit)) if index >= offset]
    if not rows:
        raise KnowledgeError("Selected input window is empty")
    if any(row.get("label") is not None for row in rows):
        raise KnowledgeError("Use unlabeled input snapshots; existing human/teacher labels must not be overwritten")
    plan = prepare_judgement(rows[0])
    manifest = {"version": VERSION, "model": model, "endpoint": endpoint,
                "prompt_hash": digest([VERSION, plan["prompt_hash"]]), "prompt_version": plan["prompt_version"],
                "max_tokens": 4096, "temperature": 0, "enable_thinking": False,
                "pairs": [{"pair_id": row["pair_id"], "input_hash": input_hash(row),
                           "request_hash": digest(qwen_prompt(row)[0])} for row in rows]}
    manifest["batch_id"] = "local_" + digest(manifest)[:32]
    directory = Path(output_dir)
    if directory.is_symlink():
        raise KnowledgeError("Output directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if _read(manifest_path) != manifest:
            raise KnowledgeError("Batch inputs/model/prompt changed; choose a different output directory")
    else:
        write_private_json(manifest_path, manifest)
    checkpoints = directory / "checkpoints"
    if checkpoints.is_symlink():
        raise KnowledgeError("Checkpoint directory must not be a symlink")
    checkpoints.mkdir(exist_ok=True, mode=0o700)
    completed = 0
    for row in rows:
        result_path = checkpoints / (row["pair_id"] + ".json")
        pending_path = checkpoints / (row["pair_id"] + ".request.json")
        if result_path.exists():
            _validated_checkpoint(result_path, row, manifest)
            completed += 1
            continue
        if pending_path.exists():
            raise KnowledgeError("A previous request has no validated result; stopped without retry. Inspect its checkpoint first")
        plan = prepare_judgement(row)
        messages, fact_aliases = qwen_prompt(row)
        original_ids = {alias: original for original, alias in fact_aliases.items()}
        write_private_json(pending_path, {"batch_id": manifest["batch_id"], "input_hash": plan["input_hash"],
                                          "started_at": datetime.now(timezone.utc).isoformat()})
        payload = {"model": model, "temperature": 0, "max_tokens": 4096,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": messages,
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "business_relevance", "strict": True, "schema": plan["response_schema"]}}}
        try:
            proposed = _request(endpoint, payload, timeout)
            if (not isinstance(proposed, dict) or not isinstance(proposed.get("company_fact_ids"), list)
                    or any(not isinstance(alias, str) or alias not in original_ids
                           for alias in proposed["company_fact_ids"])):
                raise KnowledgeError("Local response cited an unknown company fact alias")
            proposed = copy.deepcopy(proposed)
            proposed["company_fact_ids"] = [original_ids[alias] for alias in proposed["company_fact_ids"]]
            judgement = validate_judgement(proposed, row)
            result = copy.deepcopy(row)
            result["label"] = {"origin": "teacher", "labeler": "local:" + model,
                               "grade": judgement["grade"], "reason": judgement["reason"], "judgement": judgement,
                               "target": "business_relevance", "label_version": plan["prompt_version"],
                               "input_hash": plan["input_hash"], "prompt_hash": manifest["prompt_hash"],
                               "request_hash": digest(messages),
                               "batch_id": manifest["batch_id"], "timeliness_is_provisional": True,
                               "independently_reviewed": False, "labeled_at": datetime.now(timezone.utc).isoformat()}
            write_private_json(result_path, result)
        except (KnowledgeError, ValueError, OSError) as error:
            write_private_json(checkpoints / (row["pair_id"] + ".error.json"),
                               {"error_type": type(error).__name__, "error": str(error)[:1000],
                                "status": "unlabeled_no_automatic_retry"})
            raise KnowledgeError("Local prelabeling stopped; pair remains unlabeled. Inspect the error/request checkpoint before restarting") from error
        completed += 1
    return {"batch_id": manifest["batch_id"], "completed": completed, "selected": len(rows),
            "independently_reviewed": False, "external_api_calls": 0}


def export_batch(batch_dir, output):
    directory = Path(batch_dir)
    manifest = _read(directory / "manifest.json")
    if manifest.get("version") != VERSION:
        raise KnowledgeError("Unknown local labeling batch")
    # Check manifest identity and require only local endpoints even during export.
    identity = dict(manifest)
    batch_id = identity.pop("batch_id", None)
    if batch_id != "local_" + digest(identity)[:32]:
        raise KnowledgeError("Local labeling manifest was modified")
    if local_endpoint(manifest["endpoint"].removesuffix("/chat/completions")) != manifest["endpoint"]:
        raise KnowledgeError("Invalid local labeling endpoint")
    rows = []
    for item in manifest["pairs"]:
        pair_id = item["pair_id"]
        if not isinstance(pair_id, str) or not pair_id.startswith("pair_") or Path(pair_id).name != pair_id:
            raise KnowledgeError("Invalid checkpoint pair ID")
        path = directory / "checkpoints" / (pair_id + ".json")
        if not path.exists():
            continue
        source = validate_pair(_read(path))
        if source["pair_id"] != pair_id or input_hash(source) != item["input_hash"]:
            raise KnowledgeError("Checkpoint differs from the selected input")
        rows.append(_validated_checkpoint(path, source, manifest))
    if not rows:
        raise KnowledgeError("No validated local predictions to export")
    return write_dataset(output, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label")
    label.add_argument("--input", required=True)
    label.add_argument("--output-dir", required=True)
    label.add_argument("--model", required=True)
    label.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    label.add_argument("--limit", type=int, default=10)
    label.add_argument("--offset", type=int, default=0)
    label.add_argument("--timeout", type=int, default=120)
    export = commands.add_parser("export")
    export.add_argument("--batch-dir", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "label":
            result = label_batch(args.input, args.output_dir, model=args.model, base_url=args.base_url,
                                 limit=args.limit, offset=args.offset, timeout=args.timeout)
        else:
            result = export_batch(args.batch_dir, args.output)
    except (KnowledgeError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
