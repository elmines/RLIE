"""Choose LLM ternary features or continuous BERT-MLP features."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np


LOGGER = logging.getLogger("rlie.bert.pipeline")
_WARNED_ALIAS = False

_FEATURE_MODES = {
    "llm": "llm",
    "bert_static": "bert_static",
    "bert_mlp": "bert_static",
    "bert_dynamic": "bert_dynamic",
}


def resolve_feature_mode(config: dict) -> str:
    global _WARNED_ALIAS
    raw = str(config.get("feature_mode", "llm") or "llm").strip().lower()
    if raw not in _FEATURE_MODES:
        raise ValueError("feature_mode must be 'llm', 'bert_static', or 'bert_dynamic'")
    if raw == "bert_mlp" and not _WARNED_ALIAS:
        _WARNED_ALIAS = True
        LOGGER.warning("feature_mode 'bert_mlp' is now 'bert_static'")
    return _FEATURE_MODES[raw]


def feature_description(mode: str) -> str:
    if mode == "bert_dynamic":
        return "fine-tuned BERT-MLP scores in [-1, 1]"
    if mode == "bert_static":
        return "static BERT-MLP scores in [-1, 1]"
    return "LLM ternary scores"


def build_mlp_bank(config: dict, task, task_root: str, train_df, val_df):
    """Return a rule model, or None when the original LLM feature mode is active."""
    mode = resolve_feature_mode(config)
    if mode == "llm":
        return None
    if mode == "bert_static":
        from rlie.bert.mlp import RuleMLPBank

        return RuleMLPBank.from_config(config, task, task_root, train_df, val_df)
    from rlie.bert.dynamic import DynamicRuleModel

    return DynamicRuleModel.from_config(config, task, task_root, train_df, val_df)


def regression_feature_matrices(feature_store, mlp_bank) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Sample-major features for logistic regression.

    LLM mode uses the ternary silver labels. Both BERT modes use continuous
    scores in [-1, 1], without a decision threshold.
    """
    if mlp_bank is None:
        train = feature_store.to_sample_feature_matrix()
        val = feature_store.to_sample_feature_matrix("val") if feature_store.val_samples is not None else None
        return train, val
    return mlp_bank.sync_and_predict(feature_store)
