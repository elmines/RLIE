"""Save and look up fixed-length BERT [CLS] embeddings."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rlie.bert.templates import sample_key, segments_for_frame, template_label


def task_slug(task_root: str) -> str:
    text = str(task_root).strip().replace("\\", "/")
    if text.endswith(".yaml") or text.endswith(".yml"):
        text = os.path.basename(text)
        text = text.rsplit(".", 1)[0]
    return text.replace("/", "_") or "task"


def embedding_dir_for_task(embeddings_root: str, task_root: str) -> str:
    return os.path.join(embeddings_root, task_slug(task_root))


def save_split(path: str, embeddings: np.ndarray, keys: List[str]) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (samples, hidden), got {embeddings.shape}")
    if len(keys) != embeddings.shape[0]:
        raise ValueError("Embedding rows and sample keys must have the same length")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        keys=np.asarray(keys, dtype="U64"),
    )


def write_meta(directory: str, payload: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class EmbeddingIndex:
    """Content-addressed [CLS] vectors for one task."""

    def __init__(
        self,
        vectors_by_key: Dict[str, np.ndarray],
        *,
        directory: str,
        model_name: Optional[str],
        pooling: str,
        hidden_size: int,
        template: str,
    ):
        self.vectors_by_key = vectors_by_key
        self.directory = directory
        self.model_name = model_name
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.template = template

    def align(self, task, frame: pd.DataFrame) -> np.ndarray:
        if frame is None or len(frame) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)
        segments = segments_for_frame(task, frame)
        rows: List[np.ndarray] = []
        missing: List[int] = []
        for idx, segment in enumerate(segments):
            vector = self.vectors_by_key.get(sample_key(segment))
            if vector is None:
                missing.append(idx)
            else:
                rows.append(vector)
        if missing:
            preview = segments[missing[0]][0][:120] if segments[missing[0]] else ""
            raise KeyError(
                f"Missing BERT embeddings for {len(missing)} of {len(frame)} samples "
                f"under {self.directory}. Example text: {preview!r}. "
                "Run `python -m rlie.bert --config <config>` for this task before training."
            )
        return np.stack(rows).astype(np.float32, copy=False)


def load_embedding_index(directory: str) -> EmbeddingIndex:
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"BERT embedding directory not found: {directory}. "
            "Run `python -m rlie.bert --config <config>` before using feature_mode: bert_mlp."
        )
    meta_path = os.path.join(directory, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"BERT embedding metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    pooling = str(meta.get("pooling", ""))
    if pooling != "cls":
        raise ValueError(f"Embedding directory {directory} uses pooling '{pooling}', expected 'cls'")

    vectors: Dict[str, np.ndarray] = {}
    hidden_size = int(meta.get("hidden_size") or 0)
    split_files = sorted(
        name for name in os.listdir(directory) if name.endswith(".npz")
    )
    if not split_files:
        raise FileNotFoundError(f"No embedding arrays found in {directory}")
    for name in split_files:
        payload = np.load(os.path.join(directory, name), allow_pickle=False)
        matrix = np.asarray(payload["embeddings"], dtype=np.float32)
        keys = [str(key) for key in payload["keys"].tolist()]
        if matrix.ndim != 2:
            raise ValueError(f"{name} embeddings must be 2D")
        if hidden_size and matrix.shape[1] != hidden_size:
            raise ValueError(
                f"{name} hidden size {matrix.shape[1]} does not match metadata hidden size {hidden_size}"
            )
        hidden_size = matrix.shape[1]
        if len(keys) != matrix.shape[0]:
            raise ValueError(f"{name} has {matrix.shape[0]} rows but {len(keys)} keys")
        for key, vector in zip(keys, matrix):
            previous = vectors.get(key)
            if previous is not None and not np.allclose(previous, vector):
                raise ValueError(f"Conflicting BERT embeddings for the same sample in {directory}")
            vectors[key] = vector
    if hidden_size <= 0:
        raise ValueError(f"Could not determine the embedding size in {directory}")
    return EmbeddingIndex(
        vectors,
        directory=directory,
        model_name=meta.get("model_name"),
        pooling=pooling,
        hidden_size=hidden_size,
        template=str(meta.get("template") or ""),
    )


def meta_payload(
    *,
    task,
    task_root: str,
    model_name: str,
    max_length: int,
    hidden_size: int,
    splits: Dict[str, int],
) -> dict:
    return {
        "task_root": task_root,
        "task_name": getattr(task, "task_name", None),
        "model_name": model_name,
        "pooling": "cls",
        "max_length": int(max_length),
        "hidden_size": int(hidden_size),
        "template": template_label(task),
        "fields": list(getattr(task, "bert_input_fields", []) or []),
        "splits": splits,
    }
