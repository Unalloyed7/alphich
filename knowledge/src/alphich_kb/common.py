"""Content addressing and immutable artifact directories."""

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


class KnowledgeError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def safe_child(root, name):
    # Manifests are data, never paths to arbitrary files or shell instructions.
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise KnowledgeError(f"Expected a basename, got {name!r}")
    result = Path(root) / name
    if result.is_symlink():
        raise KnowledgeError(f"Symlinks are not allowed: {result}")
    return result


@contextmanager
def artifact_dir(destination):
    destination = Path(destination)
    if destination.exists():
        raise KnowledgeError(f"Output exists; select a new directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield stage
        if destination.exists():
            raise KnowledgeError(f"Output appeared during build: {destination}")
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def seal(directory, metadata):
    metadata = dict(metadata)
    metadata["outputs"] = {
        str(p.relative_to(directory)): file_hash(p)
        for p in sorted(Path(directory).rglob("*")) if p.is_file() and p.name != "manifest.json"
    }
    write_json(Path(directory) / "manifest.json", metadata)


def verify_artifacts(directory):
    manifest = read_json(Path(directory) / "manifest.json")
    for filename, expected in manifest["outputs"].items():
        path = Path(directory) / filename
        if Path(filename).is_absolute() or ".." in Path(filename).parts or not path.is_file():
            raise KnowledgeError(f"Missing/invalid artifact: {filename}")
        if file_hash(path) != expected:
            raise KnowledgeError(f"Artifact was modified: {filename}")
    return manifest
