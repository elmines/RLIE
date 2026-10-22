"""Precompute BERT [CLS] embeddings for RLIE tasks.

This step is separate from ``python -m rlie.main``. It writes one embedding per
dataset sample. Training then loads those vectors when ``feature_mode`` is
``bert_static``. ``bert_dynamic`` fine-tunes BERT during training and does not
use this cache.

Example::

    python -m rlie.bert --config configs/default.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
from rlie.utils.logger_config import LoggerConfig
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rlie.bert.embed import embed_task
from rlie.datasets.data_loader import TaskDataLoader, load_config


LOGGER = LoggerConfig.get_logger("rlie.bert")


def _task_roots(config: dict, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    roots = config.get("data_roots")
    if roots:
        if not isinstance(roots, list):
            raise ValueError("config.data_roots must be a list")
        return [str(root) for root in roots]
    single = config.get("data_root")
    if not single:
        raise ValueError("No task specified. Pass --task or set data_roots in the config.")
    return [str(single)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Encode each dataset sample as a fixed-length BERT [CLS] embedding."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Experiment config used to find tasks and BERT settings.")
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Task id to embed, such as real/headline_binary. Repeat to embed several tasks. Defaults to data_roots.",
    )
    parser.add_argument("--model", default=None, help="Hugging Face BERT model name. Defaults to bert.model_name.")
    parser.add_argument("--device", default=None, help="Device for the BERT encoder, such as cuda:0 or cpu.")
    parser.add_argument("--output", default=None, help="Directory root for saved embeddings. Defaults to bert.embeddings_dir.")
    parser.add_argument("--max-length", type=int, default=None, help="Tokenizer max length. Defaults to bert.max_length.")
    parser.add_argument("--batch-size", type=int, default=None, help="Encoding batch size. Defaults to bert.batch_size.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    bert_cfg = config.get("bert") or {}
    model_name = args.model or bert_cfg.get("model_name") or "bert-base-uncased"
    device = args.device or bert_cfg.get("device") or "cpu"
    output_root = args.output or bert_cfg.get("embeddings_dir") or "embeddings"
    max_length = int(args.max_length or bert_cfg.get("max_length") or 512)
    batch_size = int(args.batch_size or bert_cfg.get("batch_size") or 16)

    for task_root in _task_roots(config, args.task):
        task_config = dict(config)
        task_config["data_root"] = task_root
        loader = TaskDataLoader(task_config)
        directory = embed_task(
            task_root=task_root,
            task_config_path=loader.task.config_path,
            output_root=output_root,
            model_name=model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
        LOGGER.info("Finished %s embeddings in %s", task_root, directory)


if __name__ == "__main__":
    main()
