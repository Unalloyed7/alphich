"""A small exact vector index; outputs can also be imported into pgvector."""

import shutil
import time
from pathlib import Path

import numpy as np

from .common import (KnowledgeError, artifact_dir, digest, file_hash, read_jsonl,
                     seal, verify_artifacts, write_json, write_jsonl)


def build_index(corpus_dir, output_dir, encoder, batch_size=4):
    if Path(output_dir).exists():
        raise KnowledgeError(f"Output exists; select a new directory: {output_dir}")
    corpus_dir = Path(corpus_dir)
    manifest = verify_artifacts(corpus_dir)
    items = read_jsonl(corpus_dir / "chunks.jsonl") + read_jsonl(corpus_dir / "evidence.jsonl")
    if not items or len({i["id"] for i in items}) != len(items):
        raise KnowledgeError("Index requires a non-empty corpus with unique evidence IDs")
    tokenizer_hash = encoder.space["files"]["tokenizer.json"]
    if manifest["config"]["tokenizer"]["tokenizer_sha256"] != tokenizer_hash:
        raise KnowledgeError("The corpus was chunked with a different tokenizer")
    started = time.perf_counter()
    vectors = encoder.encode_passages([i["embedding_text"] for i in items], batch_size=batch_size)
    if vectors.shape != (len(items), encoder.dimension) or not np.isfinite(vectors).all():
        raise KnowledgeError("Invalid embedding matrix")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise KnowledgeError("Index vectors must be L2 normalized")
    elapsed = time.perf_counter() - started
    with artifact_dir(output_dir) as stage:
        np.save(stage / "vectors.npy", vectors, allow_pickle=False)
        write_jsonl(stage / "items.jsonl", items)
        write_jsonl(stage / "embedding_records.jsonl", [
            {"owner_id": item["id"], "owner_version": item["text_version"],
             "embedding_space_id": encoder.space_id, "input_hash": digest(item["embedding_text"]), "row": row}
            for row, item in enumerate(items)
        ])
        for name in ("documents.jsonl", "pages.jsonl", "claims.jsonl"):
            shutil.copyfile(corpus_dir / name, stage / name)
        write_json(stage / "metrics.json", {"encoding_seconds": elapsed, "items": len(items),
                                            "batch_size": batch_size, "dimension": encoder.dimension})
        version = digest({"corpus_version": manifest["version"], "space_id": encoder.space_id,
                          "vectors_hash": file_hash(stage / "vectors.npy"), "ids": [i["id"] for i in items]})
        seal(stage, {"kind": "bank_retrieval_index", "schema_version": 1, "version": version,
                     "corpus_version": manifest["version"], "embedding_space_id": encoder.space_id,
                     "embedding_space": encoder.space, "items": len(items), "dimension": encoder.dimension,
                     "chunker_config": manifest["config"]})
    return {"version": version, "items": len(items), "dimension": encoder.dimension,
            "encoding_seconds": elapsed, "output_dir": str(output_dir)}
