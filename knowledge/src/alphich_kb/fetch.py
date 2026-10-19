"""Download only pinned source files and language/tokenizer assets."""

import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from .common import KnowledgeError, file_hash, read_json
from .extract import validate_corpus


def download(url, target, sha256):
    if urlparse(url).scheme != "https":
        raise KnowledgeError("Source URL must use HTTPS")
    target = Path(target)
    if target.exists():
        if file_hash(target) != sha256:
            raise KnowledgeError(f"Existing download differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=target.parent) as tmp:
        path = Path(tmp) / "source"
        result = subprocess.run(["curl", "--location", "--fail", "--silent", "--show-error",
                                 "--max-time", "300", "--proto", "=https", "--proto-redir", "=https",
                                 url, "--output", str(path)], capture_output=True, text=True)
        if result.returncode or not path.is_file():
            raise KnowledgeError(f"Download failed: {url}: {result.stderr[-500:]}")
        if file_hash(path) != sha256:
            raise KnowledgeError(f"Downloaded checksum differs: {target.name}")
        if target.exists():
            raise KnowledgeError(f"Download target appeared: {target}")
        path.rename(target)


def fetch(corpus_path, assets_path, pdf_dir, model_dir):
    corpus = validate_corpus(corpus_path)
    for doc in corpus["documents"]:
        download(doc["download_url"], Path(pdf_dir) / doc["file"], doc["sha256"])
    for asset in read_json(assets_path)["assets"]:
        relative = Path(asset["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise KnowledgeError("Asset path must stay inside model directory")
        download(asset["url"], Path(model_dir) / relative, asset["sha256"])
    return {"documents": len(corpus["documents"]), "status": "checksums_verified"}
