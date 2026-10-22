"""Parallel, resumable corpus jobs with one private database per host group.

Groups that share any allowed hostname are joined before scheduling. Thus two
workers cannot evade a publisher's persisted request interval. There are no
paid models, production writes, inferred labels or unbounded background loops.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from array import array
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

from ..common import KnowledgeError, canonical, digest
from .corpus import SOURCE_SCHEMA, collect, iter_articles, load_sources, _signature


JOB_SCHEMA = "alphich-corpus-job-v1"


def _write(path, value, *, replace=False):
    path = Path(path)
    if not replace and path.exists():
        if path.is_symlink() or json.loads(path.read_text(encoding="utf-8")) != value:
            raise KnowledgeError("Frozen corpus job artifact differs; create a new job directory")
        return  # An initialization interrupted before job.json can resume safely.
    handle, staging = tempfile.mkstemp(prefix=".job-", dir=path.parent)
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            output.write(canonical(value) + "\n")
        if replace:
            os.replace(staging, path)
        else:
            os.link(staging, path)
    finally:
        if os.path.exists(staging):
            os.unlink(staging)


def _groups(sources):
    groups = []
    for source in sources:
        matching = [group for group in groups if group["hosts"] & set(source["allowed_hosts"])]
        if not matching:
            groups.append({"hosts": set(source["allowed_hosts"]), "source_ids": {source["id"]}})
        else:
            first = matching[0]
            first["hosts"].update(source["allowed_hosts"])
            first["source_ids"].add(source["id"])
            for other in matching[1:]:
                first["hosts"].update(other["hosts"])
                first["source_ids"].update(other["source_ids"])
                groups.remove(other)
    return sorted([{"id": "group_" + digest(sorted(group["hosts"]))[:16],
                    "hosts": sorted(group["hosts"]), "source_ids": sorted(group["source_ids"])}
                   for group in groups], key=lambda group: group["id"])


def _job(path):
    root = Path(path)
    value = json.loads((root / "job.json").read_text(encoding="utf-8"))
    if value.get("schema_version") != JOB_SCHEMA or not isinstance(value.get("groups"), list):
        raise KnowledgeError("Unsupported corpus job")
    for group in value["groups"]:
        if not isinstance(group, dict) or not re.fullmatch(r"group_[0-9a-f]{16}", group.get("id", "")):
            raise KnowledgeError("Invalid corpus job group ID")
    return root, value


def _prepare(manifest_path, root, goal, collection_window):
    sources = load_sources(manifest_path)
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    fingerprint = digest(raw)
    groups = _groups(sources)
    allowed = {source["id"] for source in sources}
    by_id = {source["id"]: source for source in raw["sources"] if source["id"] in allowed}
    if (root / "job.json").exists():
        _, existing = _job(root)
        if (existing["source_manifest_hash"] != fingerprint or existing["goal"] != goal
                or existing.get("collection_window", {"published_after": None, "published_before": None}) != collection_window):
            raise KnowledgeError("Resume requires the frozen source manifest, original goal and date window; create a new job for changes")
        frozen_root = root / "sources.json"
        if (not frozen_root.is_file() or frozen_root.is_symlink()
                or digest(json.loads(frozen_root.read_text(encoding="utf-8"))) != fingerprint
                or existing["groups"] != groups):
            raise KnowledgeError("Frozen corpus source manifest or host groups changed")
        for group in groups:
            expected = dict(raw)
            expected["sources"] = [by_id[sid] for sid in group["source_ids"]]
            frozen_group = root / (group["id"] + ".sources.json")
            if (not frozen_group.is_file() or frozen_group.is_symlink()
                    or json.loads(frozen_group.read_text(encoding="utf-8")) != expected):
                raise KnowledgeError("Frozen corpus group manifest changed; restore the original artifact or create a new job")
        return existing
    _write(root / "sources.json", raw)
    for group in groups:
        manifest = dict(raw)
        manifest["sources"] = [by_id[sid] for sid in group["source_ids"]]
        _write(root / (group["id"] + ".sources.json"), manifest)
    job = {"schema_version": JOB_SCHEMA, "source_manifest_hash": fingerprint, "goal": goal, "groups": groups,
           "collection_window": collection_window,
           "created_at_epoch": time.time(), "labels_created": 0}
    _write(root / "job.json", job)
    return job


@contextmanager
def _lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or (root / "job.lock").is_symlink():
        raise KnowledgeError("Corpus job directory/lock must not be symlinks")
    handle = os.open(root / "job.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise KnowledgeError("Another collector invocation is using this job") from error
        yield
    finally:
        os.close(handle)


def _rows(database, sql):
    if not database.exists():
        return []
    if database.is_symlink():
        raise KnowledgeError("Corpus group database must not be a symlink")
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql)]
    except sqlite3.OperationalError as error:
        if "no such table" in str(error):
            return []  # A worker has created its DB but not yet committed its schema.
        raise
    finally:
        connection.close()


def job_summary(run_dir):
    """Read committed progress without taking workers' exclusive corpus locks."""
    root, job = _job(run_dir)
    seen_urls, seen_content = set(), set()
    source_counts, groups, raw_count = {}, [], 0
    for group in job["groups"]:
        database = root / (group["id"] + ".sqlite")
        articles = _rows(database, "SELECT url,content_hash,source_id FROM articles")
        queue = {row["state"]: row["count"] for row in _rows(database, "SELECT state,COUNT(*) AS count FROM queue GROUP BY state")}
        for article in articles:
            raw_count += 1
            if article["url"] in seen_urls or article["content_hash"] in seen_content:
                continue
            seen_urls.add(article["url"])
            seen_content.add(article["content_hash"])
            source_counts[article["source_id"]] = source_counts.get(article["source_id"], 0) + 1
        groups.append({"id": group["id"], "articles": len(articles), "queue": queue})
    return {"schema_version": JOB_SCHEMA, "goal": job["goal"], "total": len(seen_content), "raw_articles": raw_count,
            "duplicates_across_groups": raw_count - len(seen_content), "sources": source_counts, "groups": groups,
            "collection_window": job.get("collection_window"),
            "labels_created": 0}


def _pending(root, group, max_source_articles):
    database = root / (group["id"] + ".sqlite")
    if not database.exists():
        return True
    counts = {row["source_id"]: row["count"] for row in _rows(database, "SELECT source_id,COUNT(*) AS count FROM articles GROUP BY source_id")}
    pending = _rows(database, "SELECT DISTINCT source_id FROM queue WHERE state='pending'")
    return any(row["source_id"] in group["source_ids"] and counts.get(row["source_id"], 0) < max_source_articles for row in pending)


def _progress(root, record, callback):
    record = {"at_epoch": time.time(), **record}
    handle = os.open(root / "progress.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(handle, (canonical(record) + "\n").encode())
    finally:
        os.close(handle)
    _write(root / "state.json", record, replace=True)
    if callback is not None:
        callback(record)


def collect_job(manifest_path, run_dir, *, goal=10_000, max_workers=4, max_runtime_seconds=3600,
                max_total_bytes=2_000_000_000, max_requests=20_000, max_articles_per_batch=100,
                batch_runtime_seconds=180, max_source_articles=2500, published_after=None, published_before=None,
                progress=None, collector=None):
    """Run bounded parallel batches and persist progress for the next invocation.

    Per-invocation HTTP/byte budgets reserve each in-flight worker's upper bound.
    Runtime is at most one hour per invocation; a 10k corpus can require several
    resumptions. collector is a local-test seam; normal jobs use corpus.collect.
    """
    values = (goal, max_workers, max_runtime_seconds, max_total_bytes, max_requests,
              max_articles_per_batch, batch_runtime_seconds, max_source_articles)
    if any(type(value) is not int or value < 1 for value in values):
        raise KnowledgeError("Corpus job budgets must be positive integers")
    if (goal > 100_000 or max_workers > 8 or max_runtime_seconds > 3600 or max_total_bytes > 2_000_000_000
            or max_total_bytes < 8_000_000 or max_requests > 100_000 or max_articles_per_batch > 1000
            or batch_runtime_seconds > 600 or max_source_articles > 10_000):
        raise KnowledgeError("Corpus job budgets exceed supported bounds")
    root = Path(run_dir)
    collector = collector or collect
    from datetime import date
    for bound in (published_after, published_before):
        if bound is not None:
            try:
                if not isinstance(bound, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", bound):
                    raise ValueError()
                date.fromisoformat(bound)
            except ValueError as failure:
                raise KnowledgeError("Corpus publication bounds must be ISO calendar dates") from failure
    if published_after and published_before and published_after > published_before:
        raise KnowledgeError("Corpus publication bounds are reversed")
    collection_window = {"published_after": published_after, "published_before": published_before}
    with _lock(root):
        job = _prepare(manifest_path, root, goal, collection_window)
        summary = job_summary(root)
        requests_used, bytes_used, articles_reserved = 0, 0, 0
        requests_reserved, bytes_reserved = 0, 0
        deadline = time.monotonic() + max_runtime_seconds
        turns = {group["id"]: 0 for group in job["groups"]}
        paused, running = set(), set()
        outcomes = []
        stop_reason = "queue_exhausted_or_source_quotas"
        _progress(root, {"event": "started", **summary, "max_workers": max_workers}, progress)
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="corpus") as executor:
            while True:
                remaining_time = int(deadline - time.monotonic())
                while len(futures) < max_workers and remaining_time > 0:
                    remaining_time = int(deadline - time.monotonic())
                    if remaining_time <= 0:
                        break
                    remaining_goal = goal - summary["total"] - articles_reserved
                    remaining_requests = max_requests - requests_used - requests_reserved
                    remaining_bytes = max_total_bytes - bytes_used - bytes_reserved
                    if remaining_goal <= 0 or remaining_requests <= 0 or remaining_bytes < 8_000_000:
                        break
                    eligible = [group for group in job["groups"] if group["id"] not in running | paused
                                and _pending(root, group, max_source_articles)]
                    if not eligible:
                        break
                    group = min(eligible, key=lambda item: (turns[item["id"]], item["id"]))
                    article_cap = min(max_articles_per_batch, remaining_goal)
                    request_cap = min(400, remaining_requests)
                    # Allocate reasonably equal byte reservations, leaving room
                    # for other workers even when the total budget is small.
                    free_slots = max(1, min(max_workers - len(futures), len(eligible)))
                    byte_cap = min(200_000_000, max(8_000_000, remaining_bytes // free_slots))
                    runtime_cap = min(batch_runtime_seconds, remaining_time)
                    future = executor.submit(collector, root / (group["id"] + ".sources.json"),
                        root / (group["id"] + ".sqlite"), goal=goal, max_articles=article_cap,
                        max_requests=request_cap, max_total_bytes=byte_cap, max_runtime_seconds=runtime_cap,
                        max_source_articles=max_source_articles,
                        published_after=published_after, published_before=published_before)
                    futures[future] = {"group_id": group["id"], "articles": article_cap, "requests": request_cap, "bytes": byte_cap}
                    running.add(group["id"])
                    articles_reserved += article_cap
                    requests_reserved += request_cap
                    bytes_reserved += byte_cap
                    turns[group["id"]] += 1
                if not futures:
                    if summary["total"] >= goal:
                        stop_reason = "corpus_goal"
                    elif time.monotonic() >= deadline:
                        stop_reason = "runtime_budget"
                    elif requests_used >= max_requests or max_total_bytes - bytes_used < 8_000_000:
                        stop_reason = "request_or_byte_budget"
                    break
                done, _ = wait(futures, timeout=min(10, max(0.1, deadline - time.monotonic())), return_when=FIRST_COMPLETED)
                for future in done:
                    reservation = futures.pop(future)
                    group_id = reservation["group_id"]
                    running.remove(group_id)
                    articles_reserved -= reservation["articles"]
                    requests_reserved -= reservation["requests"]
                    bytes_reserved -= reservation["bytes"]
                    try:
                        result = future.result()
                        requests_used += result["requests"]
                        bytes_used += result.get("byte_budget_used", result.get("response_bytes", reservation["bytes"]))
                        if result["stop_reason"] in {"source_quotas_or_queue_exhausted", "queue_exhausted", "corpus_goal", "publisher_cooldown"}:
                            paused.add(group_id)
                        # Archive queues can be huge after the recent window is
                        # exhausted. Stop scanning a publisher for this round
                        # when a complete batch yields only rejected material.
                        if result.get("new", 0) == 0 and result.get("quarantined", 0) >= 100:
                            paused.add(group_id)
                        if result["requests"] == 0 and result.get("new", 0) == 0:
                            paused.add(group_id)
                    except Exception as failure:
                        # A failed worker may already have made requests. Charge
                        # its complete reservation, and do not loop on failure.
                        requests_used += reservation["requests"]
                        bytes_used += reservation["bytes"]
                        paused.add(group_id)
                        result = {"error": type(failure).__name__, "stop_reason": "worker_failed"}
                    outcomes.append({"group_id": group_id, "result": result})
                    summary = job_summary(root)
                    _progress(root, {"event": "batch_completed", **summary, "group_id": group_id,
                        "batch": result, "requests_used": requests_used, "byte_budget_used": bytes_used}, progress)
        final = {**job_summary(root), "stop_reason": stop_reason, "requests": requests_used,
                 "byte_budget_used": bytes_used, "batches": len(outcomes),
                 "resume_manifest": str(root / "sources.json"), "run_dir": str(root)}
        _progress(root, {"event": "stopped", **final}, progress)
        return final


class _Components:
    def __init__(self):
        self.parents = []
        self.ranks = []

    def add(self):
        index = len(self.parents)
        self.parents.append(index)
        self.ranks.append(0)
        return index

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first == second:
            return
        if self.ranks[first] < self.ranks[second]:
            first, second = second, first
        self.parents[second] = first
        if self.ranks[first] == self.ranks[second]:
            self.ranks[first] += 1


def _similar_signatures(first, second):
    """Jaccard of sorted compact hash samples, without allocating large sets."""
    left = right = overlap = 0
    while left < len(first) and right < len(second):
        if first[left] == second[right]:
            overlap += 1
            left += 1
            right += 1
        elif first[left] < second[right]:
            left += 1
        else:
            right += 1
    return overlap / max(1, len(first) + len(second) - overlap) >= 0.82


def _frozen_articles(database, records):
    """Read only first-pass IDs, releasing SQLite locks before every yield."""
    if not database.is_file() or database.is_symlink():
        raise KnowledgeError("Frozen corpus database is unavailable")
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        for offset in range(0, len(records), 128):
            batch = records[offset:offset + 128]
            placeholders = ",".join("?" for _ in batch)
            rows = db.execute(f"SELECT * FROM articles WHERE article_id IN ({placeholders})",
                              [record["article_id"] for record in batch]).fetchall()
            by_id = {row["article_id"]: row for row in rows}
            for record in batch:
                row = by_id.get(record["article_id"])
                if (row is None or row["url"] != record["url"] or row["content_hash"] != record["content_hash"]
                        or row["source_id"] != record["source_id"]):
                    raise KnowledgeError("An immutable corpus snapshot changed during export")
                value = dict(row)
                value["source_metadata"] = json.loads(value["source_metadata"])
                value.pop("signature")
                value.pop("title_norm")
                value["source_version"] = value["content_hash"]
                value["cluster_id"] = record["export_cluster_id"]
                yield value
    finally:
        db.close()


def iter_job_articles(run_dir):
    """Two-pass export with connected components of all detected reprint links.

    The first pass retains IDs and compact uint64 shingle samples, never full
    article bodies. The second reads exactly those immutable IDs, so collection
    can continue during export. Source-cluster, identical-title and every
    matching sampled-shingle link are unioned, including links that bridge two
    existing components. Final component IDs are independent of iteration order.

    Candidate anchors and bottom-128 signatures are approximate duplicate
    detection. They can miss paraphrases and are not independently verified gold
    clusters or a proof that every possible train/test text leak is eliminated.
    """
    root, job = _job(run_dir)
    records, components = [], _Components()
    urls, contents, titles, anchors, local_clusters = {}, {}, {}, {}, {}
    for group in job["groups"]:
        database = root / (group["id"] + ".sqlite")
        if not database.exists():
            continue
        for article in iter_articles(database):
            title = " ".join(re.findall(r"\w+", article["title"].casefold().replace("ё", "е")))
            known_url, known_content = urls.get(article["url"]), contents.get(article["content_hash"])
            index = known_content if known_content is not None else known_url
            if index is None:
                if len(records) >= 100_000:
                    raise KnowledgeError("Corpus export exceeds the 100000-article metadata bound")
                index = components.add()
                # Packed uint64s need about 10MB for 10k x 128 hashes, rather
                # than retaining 1.28 million Python strings or all text bodies.
                signature = array("Q", sorted(int(value, 16) for value in _signature(article["text"])))
                records.append({"database": database, "article_id": article["article_id"], "url": article["url"],
                                "content_hash": article["content_hash"], "source_id": article["source_id"],
                                "signature": signature})
                candidates = {candidate for anchor in signature[:8] for candidate in anchors.get(anchor, [])}
                for candidate in candidates:
                    if (components.find(index) != components.find(candidate)
                            and _similar_signatures(signature, records[candidate]["signature"])):
                        components.union(index, candidate)
                for anchor in signature[:8]:
                    anchors.setdefault(anchor, []).append(index)
            else:
                if known_url is not None:
                    components.union(index, known_url)
                if known_content is not None:
                    components.union(index, known_content)
            # Duplicate bodies can still provide a second title/source-cluster
            # link. Keep those links even though the body is exported only once.
            if title in titles:
                components.union(index, titles[title])
            titles[title] = index
            local_key = (group["id"], article["cluster_id"])
            if local_key in local_clusters:
                components.union(index, local_clusters[local_key])
            local_clusters[local_key] = index
            urls[article["url"]] = index
            contents[article["content_hash"]] = index
    members = {}
    for index, record in enumerate(records):
        members.setdefault(components.find(index), []).append(record["content_hash"])
    cluster_ids = {component: "cluster_" + digest({"method": "detected-reprint-components-v2", "members": sorted(hashes)})[:24]
                   for component, hashes in members.items()}
    by_database = {}
    for index, record in enumerate(records):
        record["export_cluster_id"] = cluster_ids[components.find(index)]
        record.pop("signature")
        by_database.setdefault(record["database"], []).append(record)
    for database, database_records in by_database.items():
        yield from _frozen_articles(database, database_records)
