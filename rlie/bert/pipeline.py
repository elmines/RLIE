"""Choose LLM ternary features or continuous BERT-MLP features."""

from __future__ import annotations

from typing import Optional

import numpy as np


def resolve_feature_mode(config: dict) -> str:
    raw = str(config.get("feature_mode", "llm") or "llm").strip().lower()
    if raw == "llm":
        return "llm"
    if raw == "bert_mlp":
        return "bert_mlp"
    raise ValueError("feature_mode must be 'llm' or 'bert_mlp'")


def build_mlp_bank(config: dict, task, task_root: str, train_df, val_df):
    """Return a rule-MLP bank, or None when the original LLM feature mode is active."""
    if resolve_feature_mode(config) != "bert_mlp":
        return None
    from rlie.bert.mlp import RuleMLPBank

    return RuleMLPBank.from_config(config, task, task_root, train_df, val_df)


def regression_feature_matrices(feature_store, mlp_bank) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Sample-major features for logistic regression.

    In the original mode these are the LLM ternary labels. In BERT-MLP mode they
    are continuous predictions in [-1, 1], without a decision threshold.
    """
    if mlp_bank is None:
        train = feature_store.to_sample_feature_matrix()
        val = feature_store.to_sample_feature_matrix("val") if feature_store.val_samples is not None else None
        return train, val
    return mlp_bank.sync_and_predict(feature_store)
