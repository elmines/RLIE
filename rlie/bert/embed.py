"""Encode dataset samples as fixed-length BERT [CLS] embeddings."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from rlie.bert.store import embedding_dir_for_task, meta_payload, save_split, write_meta
from rlie.bert.templates import sample_key, segments_for_frame
from rlie.datasets.tasks import BaseTask


LOGGER = logging.getLogger("rlie.bert.embed")


def load_split_frame(data_dir: str, relative_path: str) -> pd.DataFrame:
    path = os.path.normpath(os.path.join(data_dir, relative_path))
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        frame = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        frame = pd.DataFrame(payload)
    else:
        raise ValueError(f"Unsupported dataset JSON in {path}")
    return frame.reset_index(drop=True)


def split_paths(task: BaseTask) -> Dict[str, str]:
    with open(task.config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    splits = {
        "train": raw["train_data_path"],
        "val": raw["val_data_path"],
        "test": raw["test_data_path"],
    }
    if raw.get("ood_data_path"):
        splits["ood"] = raw["ood_data_path"]
    return splits


def tokenizer_arguments(
    batch_segments: Sequence[Sequence[str]],
    sep_token: str = "[SEP]",
) -> tuple[List[str], Optional[List[str]]]:
    """Build the text pair that the tokenizer turns into ``[CLS] ... [SEP] ...``."""
    if not batch_segments:
        return [], None
    width = len(batch_segments[0])
    if width <= 0:
        raise ValueError("BERT template produced an empty segment list")
    if any(len(item) != width for item in batch_segments):
        raise ValueError("Every sample must have the same number of BERT segments")
    first = [str(item[0]) for item in batch_segments]
    if width == 1:
        return first, None
    if width == 2:
        return first, [str(item[1]) for item in batch_segments]
    tails = [f" {sep_token} ".join(str(part) for part in item[1:]) for item in batch_segments]
    return first, tails


def pool_cls(last_hidden_state) -> "torch.Tensor":
    """Return the hidden state at the [CLS] position (index 0)."""
    import torch

    if not isinstance(last_hidden_state, torch.Tensor) or last_hidden_state.ndim != 3:
        raise ValueError("BERT last_hidden_state must be a tensor of shape (batch, sequence, hidden)")
    return last_hidden_state[:, 0, :]


def encode_cls_batches(
    segments: Sequence[Sequence[str]],
    *,
    tokenizer,
    model,
    device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    import torch
    from tqdm.auto import tqdm

    if not segments:
        hidden = int(getattr(model.config, "hidden_size", 0) or 0)
        return np.zeros((0, hidden), dtype=np.float32)

    model.eval()
    model.to(device)
    outputs: List[np.ndarray] = []
    batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        for start in tqdm(range(0, len(segments), batch_size), desc="BERT [CLS]", unit="batch"):
            batch = list(segments[start : start + batch_size])
            text, text_pair = tokenizer_arguments(batch, sep_token=tokenizer.sep_token or "[SEP]")
            tokenizer_kwargs = {
                "truncation": True,
                "padding": True,
                "max_length": int(max_length),
                "return_tensors": "pt",
            }
            if text_pair is None:
                encoded = tokenizer(text, **tokenizer_kwargs)
            else:
                encoded = tokenizer(text, text_pair, **tokenizer_kwargs)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            cls = pool_cls(hidden).detach().cpu().numpy().astype(np.float32, copy=False)
            outputs.append(cls)
    return np.concatenate(outputs, axis=0)


def embed_task(
    *,
    task_root: str,
    task_config_path: str,
    output_root: str,
    model_name: str,
    device: str,
    max_length: int,
    batch_size: int,
) -> str:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from rlie.bert.mlp import resolve_device

    task = BaseTask(config_path=task_config_path, extract_label=lambda value: value)
    directory = embedding_dir_for_task(output_root, task_root)
    os.makedirs(directory, exist_ok=True)
    torch_device = resolve_device(device)
    LOGGER.info(
        "Embedding %s with %s on %s (max_length=%d, template=%s)",
        task_root,
        model_name,
        torch_device,
        max_length,
        task.task_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    split_counts: Dict[str, int] = {}
    hidden_size: Optional[int] = None
    logged_example = False
    encoded_files: Dict[str, tuple[np.ndarray, List[str]]] = {}

    for split_name, relative_path in split_paths(task).items():
        absolute = os.path.normpath(os.path.join(task.data_dir, relative_path))
        if not os.path.exists(absolute):
            if split_name == "ood":
                LOGGER.info("Skipping missing OOD split: %s", absolute)
                continue
            raise FileNotFoundError(f"Dataset split '{split_name}' not found: {absolute}")
        cached = encoded_files.get(absolute)
        if cached is None:
            frame = load_split_frame(task.data_dir, relative_path)
            segments = segments_for_frame(task, frame)
            if not logged_example and segments:
                text, text_pair = tokenizer_arguments(segments[:1], sep_token=tokenizer.sep_token or "[SEP]")
                preview_kwargs = {"truncation": True, "max_length": int(max_length), "return_tensors": "pt"}
                if text_pair is None:
                    preview = tokenizer(text, **preview_kwargs)
                else:
                    preview = tokenizer(text, text_pair, **preview_kwargs)
                decoded = tokenizer.decode(preview["input_ids"][0], skip_special_tokens=False)
                LOGGER.info("BERT input example (%s): %s", split_name, decoded)
                logged_example = True
            matrix = encode_cls_batches(
                segments,
                tokenizer=tokenizer,
                model=model,
                device=torch_device,
                max_length=max_length,
                batch_size=batch_size,
            )
            keys = [sample_key(segment) for segment in segments]
            encoded_files[absolute] = (matrix, keys)
        matrix, keys = encoded_files[absolute]
        if matrix.size and (hidden_size is None):
            hidden_size = int(matrix.shape[1])
        save_split(os.path.join(directory, f"{split_name}.npz"), matrix, keys)
        split_counts[split_name] = int(matrix.shape[0])
        LOGGER.info("Saved %s %s embeddings to %s", split_counts[split_name], split_name, directory)

    if hidden_size is None:
        hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    write_meta(
        directory,
        meta_payload(
            task=task,
            task_root=task_root,
            model_name=model_name,
            max_length=max_length,
            hidden_size=hidden_size,
            splits=split_counts,
        ),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return directory
