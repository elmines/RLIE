"""Task-specific BERT input templates.

Each template returns one or more text segments. The embedder encodes them with
the BERT tokenizer, which inserts the special tokens:

- one segment: ``[CLS] segment [SEP]``
- two segments: ``[CLS] segment_1 [SEP] segment_2 [SEP]``

The embedding kept for downstream models is the hidden state at the ``[CLS]``
position. Register a function to customize a task, or set ``bert_input.fields``
in that task's ``task.yaml`` to use the default segment template.
"""

from __future__ import annotations

import hashlib
from rlie.utils.logger_config import LoggerConfig
import math
from typing import Callable, Dict, List, Sequence

import pandas as pd


LOGGER = LoggerConfig.get_logger("rlie.bert.templates")

BertBuilder = Callable[[pd.Series], List[str]]
TASK_BERT_BUILDERS: Dict[str, BertBuilder] = {}
_WARNED_MISMATCH: set[str] = set()


def register_bert_template(task_name: str, fields: Sequence[str]):
    """Register the function that builds BERT segments for ``task_name``."""

    def decorator(fn: BertBuilder) -> BertBuilder:
        fn.fields = tuple(fields)  # type: ignore[attr-defined]
        TASK_BERT_BUILDERS[task_name] = fn
        return fn

    return decorator


def _text(row: pd.Series, field: str) -> str:
    if isinstance(row, pd.Series):
        if field not in row.index:
            raise KeyError(f"BERT template field '{field}' is missing from the sample")
        value = row[field]
    else:
        if field not in row:
            raise KeyError(f"BERT template field '{field}' is missing from the sample")
        value = row[field]
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _fields_builder(fields: Sequence[str]) -> BertBuilder:
    field_list = tuple(fields)

    def builder(row: pd.Series) -> List[str]:
        return [_text(row, field) for field in field_list]

    builder.fields = field_list  # type: ignore[attr-defined]
    return builder


@register_bert_template("headline_binary", ("headline_1", "headline_2"))
def headline_binary_segments(row: pd.Series) -> List[str]:
    """[CLS] headline_1 [SEP] headline_2 [SEP]"""
    return [_text(row, "headline_1"), _text(row, "headline_2")]


@register_bert_template("retweet", ("first_tweet", "second_tweet"))
def retweet_segments(row: pd.Series) -> List[str]:
    """[CLS] first_tweet [SEP] second_tweet [SEP]"""
    return [_text(row, "first_tweet"), _text(row, "second_tweet")]


@register_bert_template("persuasive_pairs", ("argument_1", "argument_2"))
def persuasive_pairs_segments(row: pd.Series) -> List[str]:
    """[CLS] argument_1 [SEP] argument_2 [SEP]"""
    return [_text(row, "argument_1"), _text(row, "argument_2")]


@register_bert_template("dreaddit", ("text",))
def dreaddit_segments(row: pd.Series) -> List[str]:
    """[CLS] text [SEP]"""
    return [_text(row, "text")]


@register_bert_template("deceptive_reviews", ("review_sentence",))
def deceptive_reviews_segments(row: pd.Series) -> List[str]:
    """[CLS] review_sentence [SEP]"""
    return [_text(row, "review_sentence")]


@register_bert_template("llamagc_detect", ("story",))
def llamagc_detect_segments(row: pd.Series) -> List[str]:
    """[CLS] story [SEP]"""
    return [_text(row, "story")]


@register_bert_template("vast", ("post", "topic"))
def vast_segments(row: pd.Series) -> List[str]:
    """[CLS] post [SEP] topic [SEP]"""
    return [_text(row, "post"), _text(row, "topic")]


def resolve_builder(task) -> BertBuilder:
    """Return the segment builder for a task, preferring a registered function."""
    builder = TASK_BERT_BUILDERS.get(task.task_name)
    yaml_fields = tuple(getattr(task, "bert_input_fields", []) or ())
    if builder is not None:
        expected = tuple(getattr(builder, "fields", ()) or ())
        if expected and yaml_fields and expected != yaml_fields and task.task_name not in _WARNED_MISMATCH:
            _WARNED_MISMATCH.add(task.task_name)
            LOGGER.warning(
                "Task %s bert_input.fields %s differs from the registered template %s; using the registered template",
                task.task_name,
                list(yaml_fields),
                list(expected),
            )
        return builder
    if not yaml_fields:
        raise ValueError(
            f"No BERT input template for task '{task.task_name}'. "
            "Register one with rlie.bert.templates.register_bert_template or set bert_input.fields in task.yaml."
        )
    return _fields_builder(yaml_fields)


def segments_for_row(task, row: pd.Series) -> List[str]:
    return list(resolve_builder(task)(row))


def segments_for_frame(task, frame: pd.DataFrame) -> List[List[str]]:
    return [segments_for_row(task, frame.iloc[idx]) for idx in range(len(frame))]


def sample_key(segments: Sequence[str]) -> str:
    """Stable id for a BERT input. Identical segments share an embedding."""
    payload = "\x1f".join(str(segment) for segment in segments)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def template_label(task) -> str:
    builder = TASK_BERT_BUILDERS.get(getattr(task, "task_name", None))
    if builder is not None and builder.__doc__:
        return builder.__doc__.strip().splitlines()[0]
    fields = tuple(getattr(task, "bert_input_fields", []) or ())
    if not fields:
        return ""
    if len(fields) == 1:
        return f"[CLS] {fields[0]} [SEP]"
    return "[CLS] " + " [SEP] ".join(fields) + " [SEP]"
