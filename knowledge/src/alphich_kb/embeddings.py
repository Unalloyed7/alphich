"""Pinned, offline E5 inference using ONNX Runtime CPU, without torch or a GPU."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from .common import KnowledgeError, digest, file_hash, read_json, safe_child


class EncoderUnavailable(KnowledgeError):
    """An operational inference failure; distinct from invalid input or stale data."""


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
        raise KnowledgeError("Embedding contains non-finite values or a zero vector")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def mean_pool(hidden, attention_mask):
    hidden = np.asarray(hidden, dtype=np.float32)
    mask = np.asarray(attention_mask, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[:2] != mask.shape or np.any(mask.sum(axis=1) == 0):
        raise KnowledgeError("Unexpected E5 hidden-state or attention-mask shape")
    pooled = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
    return normalize(pooled)


class E5Encoder:
    def __init__(self, model_dir="data/models/e5-base", model_manifest="knowledge/embedding-model.json", threads=2):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer, __version__ as tokenizer_version
        except ImportError as error:
            raise EncoderUnavailable("Install knowledge/requirements-rag.lock for E5 inference") from error
        if not 1 <= threads <= 16:
            raise KnowledgeError("Encoder threads must be 1..16")
        self.config = read_json(model_manifest)
        for item in self.config["files"]:
            path = safe_child(model_dir, item["path"])
            if not path.is_file():
                raise EncoderUnavailable(f"Missing E5 asset: {item['path']}; run fetch-encoder")
            if file_hash(path) != item["sha256"]:
                raise KnowledgeError(f"E5 checksum mismatch: {item['path']}")
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.max_tokens = self.config["max_tokens"]
        self.dimension = self.config["dimension"]
        self.space = {
            "model_id": self.config["model_id"], "revision": self.config["revision"],
            "files": {f["path"]: f["sha256"] for f in self.config["files"]}, "format": self.config["format"],
            "dimension": self.dimension, "max_tokens": self.max_tokens,
            "pooling": "attention_mask_mean", "normalization": "l2", "prefixes": ["query: ", "passage: "],
            "runtime": f"onnxruntime=={ort.__version__}", "tokenizer_runtime": f"tokenizers=={tokenizer_version}",
            "numpy": np.__version__, "provider": "CPUExecutionProvider", "implementation": "e5-onnx-v1",
        }
        self.space_id = f"embedding_space_{digest(self.space)}"
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        model_file = next(f["path"] for f in self.config["files"] if f["path"].endswith(".onnx"))
        try:
            self.session = ort.InferenceSession(str(Path(model_dir) / model_file), sess_options=options,
                                                providers=["CPUExecutionProvider"])
        except Exception as error:
            raise EncoderUnavailable(f"Cannot load the pinned ONNX graph: {type(error).__name__}") from error
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name
        if not {"input_ids", "attention_mask"}.issubset(self.inputs):
            raise KnowledgeError("E5 graph is missing input_ids or attention_mask")

    def _encode(self, texts, prefix):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(t, str) or not t.startswith(prefix) or not t[len(prefix):].strip() for t in texts):
            raise KnowledgeError(f"Each prepared E5 input must contain exactly one {prefix!r} prefix")
        if any(t[len(prefix):].startswith(("query: ", "passage: ")) for t in texts):
            raise KnowledgeError("Duplicate E5 prefix")
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=True)
        lengths = [len(e.ids) for e in encodings]
        if max(lengths) > self.max_tokens:
            raise KnowledgeError(f"E5 input exceeds {self.max_tokens} tokens; split it instead of truncating")
        width = max(lengths)
        pad_id = self.tokenizer.token_to_id("<pad>")
        ids = np.full((len(texts), width), pad_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for row, encoding in enumerate(encodings):
            ids[row, :lengths[row]] = encoding.ids
            mask[row, :lengths[row]] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        try:
            hidden = self.session.run([self.output_name], feed)[0]
        except Exception as error:
            raise EncoderUnavailable(f"E5 inference failed: {type(error).__name__}") from error
        vectors = mean_pool(hidden, mask)
        if vectors.shape != (len(texts), self.dimension):
            raise KnowledgeError("Unexpected E5 embedding dimension")
        return vectors

    def encode_passages(self, prepared_texts, batch_size=4):
        if not 1 <= batch_size <= 32:
            raise KnowledgeError("Embedding batch size must be 1..32")
        batches = [self._encode(prepared_texts[i:i + batch_size], "passage: ")
                   for i in range(0, len(prepared_texts), batch_size)]
        return np.concatenate(batches) if batches else np.empty((0, self.dimension), dtype=np.float32)

    def count_tokens(self, prepared_text):
        return len(self.tokenizer.encode(prepared_text, add_special_tokens=True).ids)

    @lru_cache(maxsize=128)
    def _cached_query(self, query):
        return self._encode(["query: " + query], "query: ")[0]

    def encode_query(self, query):
        if not isinstance(query, str) or not query.strip() or query.startswith(("query: ", "passage: ")):
            raise KnowledgeError("Pass non-empty raw query text without an E5 prefix")
        return self._cached_query(query.strip()).copy()
