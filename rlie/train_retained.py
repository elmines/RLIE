"""Train BERT and one MLP per retained hypothesis from an LLM run.

The source run must be ``feature_mode: llm``. Only hypotheses whose logistic
regression weight is nonzero in the best round are kept. Their ternary silver
labels supervise the rule MLPs, and a new logistic regression is fit on the
MLP features with the scikit-learn config.

Example::

    python -m rlie.train_retained --config configs/default.yaml --run-dir outputs/run_.../real_headline_binary --feature-mode bert_dynamic
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rlie.utils.logger_config import LoggerConfig
from rlie.bert.pipeline import build_mlp_bank, resolve_feature_mode
from rlie.core.feature_manager import RuleFeatureStore
from rlie.core.metrics import compute_accuracy_and_macro_f1
from rlie.core.modeling import SoftmaxTrainer
from rlie.core.reporting import load_run_samples, save_dataset_csv, save_json, save_rule_snapshot
from rlie.datasets.data_loader import TaskDataLoader, apply_external_configs, load_config
from rlie.datasets.labels import map_labels


LOGGER = LoggerConfig.get_logger("rlie.train_retained")
ACTIVE_WEIGHT = 1e-12


def _parse_repeat_index(run_dir: str) -> int:
    match = re.search(r"_repeat(\d+)", os.path.basename(os.path.abspath(run_dir)))
    if match:
        return max(int(match.group(1)) - 1, 0)
    return 0


def _resolve_task_root(config: dict, task_root: Optional[str]) -> str:
    if task_root:
        return task_root
    single = config.get("data_root")
    if isinstance(single, str) and single.strip():
        return single
    roots = config.get("data_roots") or []
    if len(roots) == 1:
        return str(roots[0])
    if len(roots) > 1:
        raise ValueError("config.data_roots contains multiple tasks; pass --task-root")
    raise ValueError("Missing task path. Set data_root or pass --task-root.")


def _load_round_payload(run_dir: str, round_id: Optional[int]) -> Tuple[int, Optional[str], dict]:
    metrics_path = os.path.join(run_dir, "final_round_metrics.json")
    recorded_mode = None
    best_round = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        recorded_mode = metrics.get("feature_mode")
        if "round" in metrics:
            best_round = int(metrics["round"])
    if recorded_mode in {"bert_static", "bert_dynamic"}:
        raise ValueError(
            f"Run {run_dir} used feature_mode '{recorded_mode}'. "
            "train_retained reads ternary silver labels from an LLM run."
        )
    chosen = best_round if round_id is None else int(round_id)
    if chosen is None:
        raise FileNotFoundError(f"No best round recorded in {metrics_path}")
    snapshot_path = os.path.join(run_dir, "rule_snapshots", f"round_{chosen:02d}_rules.json")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Rule snapshot does not exist: {snapshot_path}")
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return chosen, None if recorded_mode is None else str(recorded_mode), payload


def read_round_feature_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return mapped labels and the sample-major rule matrix from a round CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Round dataset does not exist: {path}")
    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError(f"{path} is missing the label column")
    rule_columns = [column for column in frame.columns if re.fullmatch(r"rule_\d+", column)]
    rule_columns.sort(key=lambda name: int(name.split("_", 1)[1]))
    labels = frame["label"].to_numpy()
    if not rule_columns:
        matrix = np.zeros((len(frame), 0), dtype=np.float64)
    else:
        matrix = frame[rule_columns].to_numpy(dtype=np.float64)
    return labels, matrix


def _as_ternary(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.rint(np.asarray(matrix, dtype=float))
    if values.size and not np.isin(values, (-1, 0, 1)).all():
        raise ValueError(f"{name} is not a ternary silver-label matrix with values -1, 0, and 1")
    return values.astype(np.int8)


def active_entries(entries: Sequence[dict], tol: float = ACTIVE_WEIGHT) -> List[dict]:
    kept = []
    for idx, entry in enumerate(entries):
        weight = float(entry.get("weight", 0.0))
        if abs(weight) <= tol:
            continue
        copied = dict(entry)
        copied["rule_index"] = int(entry.get("rule_index", idx))
        copied["weight"] = weight
        kept.append(copied)
    return kept


def _metric_dicts(entry: dict, prefix: str) -> dict:
    return {
        "accuracy": entry.get(f"{prefix}_accuracy"),
        "f1": entry.get(f"{prefix}_f1"),
        "coverage": entry.get(f"{prefix}_coverage"),
    }


def build_retained_store(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    entries: Sequence[dict],
    train_features: np.ndarray,
    val_features: Optional[np.ndarray],
    *,
    label_column: str,
    source_round: int,
) -> RuleFeatureStore:
    """Keep nonzero-weight rules and the silver labels in those columns."""
    selected = active_entries(entries)
    if not selected:
        raise ValueError("Best round has no hypotheses with a nonzero logistic-regression weight")
    train_ternary = _as_ternary(train_features, "Training rules")
    if train_ternary.shape != (len(train_df), len(entries)):
        raise ValueError(
            f"Training silver labels have shape {train_ternary.shape}; "
            f"expected {(len(train_df), len(entries))}"
        )
    val_ternary = None
    if val_df is not None and len(val_df) > 0:
        if val_features is None:
            raise ValueError("Validation samples are present but the validation rule matrix is missing")
        val_ternary = _as_ternary(val_features, "Validation rules")
        if val_ternary.shape != (len(val_df), len(entries)):
            raise ValueError(
                f"Validation silver labels have shape {val_ternary.shape}; "
                f"expected {(len(val_df), len(entries))}"
            )
    indices = [int(entry["rule_index"]) for entry in selected]
    if any(index < 0 or index >= train_ternary.shape[1] for index in indices):
        raise ValueError("Snapshot rule_index does not match a column of the silver-label matrix")
    train_rules = train_ternary[:, indices].T
    val_rules = None if val_ternary is None else val_ternary[:, indices].T
    store = RuleFeatureStore(
        samples=train_df,
        label_column=label_column,
        val_samples=None if val_df is None or len(val_df) == 0 else val_df,
    )
    store.add_rules(
        [str(entry.get("rule", "")) for entry in selected],
        train_rules.tolist(),
        int(source_round),
        val_vectors=None if val_rules is None else val_rules.tolist(),
        train_metrics=[_metric_dicts(entry, "train") for entry in selected],
        val_metrics=[_metric_dicts(entry, "val") for entry in selected],
    )
    return store


def _frames_for_run(run_dir: str, config: dict, task_root: str, repeat_index: int):
    train_df, val_df, _test_df = load_run_samples(run_dir)
    if train_df is not None and len(train_df) > 0:
        if val_df is not None and len(val_df) == 0:
            val_df = None
        return train_df.reset_index(drop=True), None if val_df is None else val_df.reset_index(drop=True), False
    from rlie.run_eval_only import _prepare_datasets

    LOGGER.warning(
        "No samples/train.json in %s; reloading the dataset. "
        "Train row order is only reliable when that file was saved with the run.",
        run_dir,
    )
    _loader, train_df, val_df, _test_df = _prepare_datasets(config, task_root, repeat_index)
    if len(val_df) == 0:
        val_df = None
    return train_df, val_df, True


def _labels_match(frame: pd.DataFrame, csv_labels: np.ndarray, label_mapping: dict, label_column: str) -> None:
    if len(frame) != len(csv_labels):
        raise ValueError(
            f"Recorded samples have {len(frame)} rows but the round dataset has {len(csv_labels)}"
        )
    mapped = np.asarray(map_labels(frame, label_mapping, label_column), dtype=int)
    saved = np.rint(np.asarray(csv_labels, dtype=float)).astype(int)
    if not np.array_equal(mapped, saved):
        raise ValueError(
            "The recorded samples are not in the same order as the round dataset. "
            "Use the samples/train.json written by the LLM run."
        )


def load_retained_store(
    run_dir: str,
    config: dict,
    *,
    task_root: str,
    repeat_index: int,
    round_id: Optional[int] = None,
    label_mapping: dict,
    label_column: str,
) -> Tuple[RuleFeatureStore, pd.DataFrame, Optional[pd.DataFrame], int]:
    chosen, _mode, payload = _load_round_payload(run_dir, round_id)
    chosen = int(chosen)
    entries = list(payload.get("rules") or [])
    train_df, val_df, reloaded = _frames_for_run(run_dir, config, task_root, repeat_index)
    train_labels, train_matrix = read_round_feature_csv(
        os.path.join(run_dir, f"round_{chosen}_train_dataset.csv")
    )
    if not reloaded:
        _labels_match(train_df, train_labels, label_mapping, label_column)
    elif len(train_df) != len(train_labels):
        raise ValueError(
            f"Reloaded training data has {len(train_df)} rows but the round dataset has {len(train_labels)}"
        )
    val_matrix = None
    if val_df is not None:
        val_labels, val_matrix = read_round_feature_csv(
            os.path.join(run_dir, f"round_{chosen}_val_dataset.csv")
        )
        if not reloaded:
            _labels_match(val_df, val_labels, label_mapping, label_column)
        elif len(val_df) != len(val_labels):
            raise ValueError(
                f"Reloaded validation data has {len(val_df)} rows but the round dataset has {len(val_labels)}"
            )
    store = build_retained_store(
        train_df,
        val_df,
        entries,
        train_matrix,
        val_matrix,
        label_column=label_column,
        source_round=chosen,
    )
    LOGGER.info(
        "Retained %d/%d hypotheses from round %d",
        len(store.rules),
        len(entries),
        chosen,
    )
    return store, train_df, val_df, chosen


def _score_split(model, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probabilities = model.predict_proba(np.asarray(features, dtype=float))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    label_space = sorted({int(value) for value in np.unique(labels)})
    accuracy, f1 = compute_accuracy_and_macro_f1(labels, predictions, labels=label_space)
    return {"accuracy": accuracy, "f1": f1}


def fit_retained(
    config: dict,
    *,
    task,
    task_root: str,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    store: RuleFeatureStore,
    train_labels: np.ndarray,
    val_labels: Optional[np.ndarray],
    output_dir: str,
    feature_mode: str,
    source_round: int,
    source_run_dir: str,
    bank=None,
) -> dict:
    """Train the configured BERT mode on the retained rules and refit logistic regression."""
    mode = resolve_feature_mode({"feature_mode": feature_mode})
    if mode not in {"bert_static", "bert_dynamic"}:
        raise ValueError("feature_mode must be 'bert_static' or 'bert_dynamic'")
    run_config = copy.deepcopy(config)
    run_config["feature_mode"] = mode
    run_config["data_root"] = task_root
    os.makedirs(output_dir, exist_ok=True)
    if bank is None:
        bank = build_mlp_bank(
            run_config,
            task,
            task_root,
            train_df,
            None if val_df is None or len(val_df) == 0 else val_df,
        )
    features, val_features = bank.sync_and_predict(store)
    result = SoftmaxTrainer(run_config).train(features, np.asarray(train_labels))
    bank.save_round(int(source_round), output_dir, store.rules)
    save_rule_snapshot(int(source_round), store, result.model, output_dir)
    save_dataset_csv(features, train_labels, os.path.join(output_dir, "train_features.csv"))
    payload = {
        "source_run_dir": os.path.abspath(source_run_dir),
        "source_round": int(source_round),
        "feature_mode": mode,
        "rules": [rule.text for rule in store.rules],
        "train": {
            "accuracy": result.accuracy,
            "f1": result.f1,
            "retained_hypotheses": result.retained_hypotheses,
            "total_hypotheses": result.total_hypotheses,
        },
        "hyperparameters": {
            "params": result.best_params,
            "selection_metric": result.selection_metric,
            "cv_best_score": result.cv_best_score,
        },
        "rule_mlp_dir": os.path.join(output_dir, "rule_mlps", f"round_{int(source_round):02d}"),
    }
    if val_features is not None and val_labels is not None and len(val_labels) > 0:
        payload["val"] = _score_split(result.model, val_features, np.asarray(val_labels))
        save_dataset_csv(val_features, val_labels, os.path.join(output_dir, "val_features.csv"))
    save_json(os.path.join(output_dir, "retained_metrics.json"), payload)
    LOGGER.info(
        "Saved retained %s model to %s (train accuracy=%.4f, active rules after refit=%d/%d)",
        mode,
        output_dir,
        result.accuracy,
        result.retained_hypotheses,
        result.total_hypotheses,
    )
    return payload


def run(
    *,
    config_path: str,
    run_dir: str,
    feature_mode: str,
    output_dir: Optional[str] = None,
    task_root: Optional[str] = None,
    round_id: Optional[int] = None,
    repeat_index: Optional[int] = None,
    bert_config: Optional[str] = None,
    head_config: Optional[str] = None,
    sklearn_config: Optional[str] = None,
) -> dict:
    config = load_config(config_path)
    if bert_config or head_config or sklearn_config:
        if bert_config:
            config["bert_config"] = bert_config
        if head_config:
            config["head_config"] = head_config
        if sklearn_config:
            config["sklearn_config"] = sklearn_config
        config = apply_external_configs(config, config_path)
    if "bert" not in config or "mlp" not in (config.get("bert") or {}):
        raise ValueError("BERT and head configs are required. Pass --bert-config and --head-config, or set them in the experiment config.")
    if "elastic_net" not in config:
        raise ValueError("Scikit-learn config is required. Pass --sklearn-config or set sklearn_config in the experiment config.")

    task_root = _resolve_task_root(config, task_root)
    if repeat_index is None:
        repeat_index = _parse_repeat_index(run_dir)
    task_config = copy.deepcopy(config)
    task_config["data_root"] = task_root
    task_config["data_roots"] = None
    task_config["data_seed"] = int(config.get("data_seed", 42)) + int(repeat_index)
    loader = TaskDataLoader(task_config)
    store, train_df, val_df, source_round = load_retained_store(
        run_dir,
        task_config,
        task_root=task_root,
        repeat_index=repeat_index,
        round_id=round_id,
        label_mapping=loader.label_mapping,
        label_column=loader.task.label_name,
    )
    train_labels = np.asarray(map_labels(train_df, loader.label_mapping, loader.task.label_name))
    val_labels = None
    if val_df is not None and len(val_df) > 0:
        val_labels = np.asarray(map_labels(val_df, loader.label_mapping, loader.task.label_name))
    destination = output_dir or os.path.join(run_dir, f"retained_{feature_mode}")
    return fit_retained(
        task_config,
        task=loader.task,
        task_root=task_root,
        train_df=train_df,
        val_df=val_df,
        store=store,
        train_labels=train_labels,
        val_labels=val_labels,
        output_dir=destination,
        feature_mode=feature_mode,
        source_round=source_round,
        source_run_dir=run_dir,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train BERT and rule MLPs on the nonzero-weight hypotheses from an LLM run."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Experiment config used to reload the task.")
    parser.add_argument("--run-dir", required=True, help="Output directory of one rlie.main repeat.")
    parser.add_argument(
        "--feature-mode",
        default="bert_dynamic",
        choices=("bert_static", "bert_dynamic"),
        help="bert_static uses cached embeddings. bert_dynamic fine-tunes BERT with the rule MLPs.",
    )
    parser.add_argument("--bert-config", default=None, help="BERT config. Defaults to bert_config in --config.")
    parser.add_argument("--head-config", default=None, help="Rule MLP config. Defaults to head_config in --config.")
    parser.add_argument("--sklearn-config", default=None, help="Logistic regression config. Defaults to sklearn_config in --config.")
    parser.add_argument("--task-root", default=None, help="Task id when the experiment config lists more than one.")
    parser.add_argument("--round", dest="round_id", type=int, default=None, help="Snapshot round. Defaults to the best round.")
    parser.add_argument("--repeat-index", type=int, default=None, help="Zero-based repeat index. Parsed from the run directory name when omitted.")
    parser.add_argument("--output", default=None, help="Directory for the BERT checkpoint and refit. Defaults to <run-dir>/retained_<feature-mode>.")
    args = parser.parse_args(argv)
    run(
        config_path=args.config,
        run_dir=args.run_dir,
        feature_mode=args.feature_mode,
        output_dir=args.output,
        task_root=args.task_root,
        round_id=args.round_id,
        repeat_index=args.repeat_index,
        bert_config=args.bert_config,
        head_config=args.head_config,
        sklearn_config=args.sklearn_config,
    )


if __name__ == "__main__":
    main()
