import argparse
import copy
import glob
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    # Support direct execution of this file during development.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.extend([
        current_dir,
        parent_dir,
    ])

import numpy as np
import pandas as pd
from rlie.llm.prompt import BasePrompt

from rlie.datasets.data_loader import TaskDataLoader, _normalize_label, load_config
from rlie.utils.eval_utils import (
    evaluate_models,
    parse_evaluation_modes,
    prepare_llm_config,
    save_json as save_json_safe,
    sanitize_identifier,
)
from rlie.datasets.labels import map_labels
from rlie.llm.rule_scorer import RuleScorer
from rlie.core.metrics import compute_accuracy_and_macro_f1
from rlie.bert.pipeline import resolve_feature_mode


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
LOGGER = logging.getLogger("rlie.run_eval_only")


def _parse_repeat_index(run_dir: str, fallback: int = 0) -> int:
    match = re.search(r"_repeat(\d+)", os.path.basename(run_dir))
    if match:
        return max(int(match.group(1)) - 1, 0)
    return fallback


def _load_rule_snapshot(run_dir: str, round_id: Optional[int] = None) -> Tuple[int, str, List[str], np.ndarray, float]:
    metrics_path = os.path.join(run_dir, "final_round_metrics.json")
    local_snapshot_dir = os.path.join(run_dir, "rule_snapshots")
    snapshot_dir = local_snapshot_dir
    best_round = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_payload = json.load(f)
        best_round = int(metrics_payload.get("round", 0))
        candidate_snapshot_dir = metrics_payload.get("rule_snapshot_dir", snapshot_dir)
        if candidate_snapshot_dir and os.path.exists(candidate_snapshot_dir):
            snapshot_dir = candidate_snapshot_dir
        elif os.path.exists(local_snapshot_dir):
            snapshot_dir = local_snapshot_dir
    if round_id is None:
        round_id = best_round

    if round_id is None:
        pattern = os.path.join(snapshot_dir, "round_*_rules.json")
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No rule snapshot files found; check that the directory exists: {snapshot_dir}")
        last = os.path.basename(candidates[-1])
        match = re.search(r"round_(\d+)_rules\.json", last)
        round_id = int(match.group(1)) if match else len(candidates) - 1

    snapshot_path = os.path.join(snapshot_dir, f"round_{round_id:02d}_rules.json")
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Rule snapshot does not exist: {snapshot_path}")

    with open(snapshot_path, "r") as f:
        payload = json.load(f)

    rules = payload.get("rules", [])
    rule_texts = [entry.get("rule", "") for entry in rules]
    weights = np.array([float(entry.get("weight", 0.0)) for entry in rules], dtype=float)
    bias = float(payload.get("bias", 0.0))
    best_round = int(payload.get("round", round_id))
    return best_round, snapshot_dir, rule_texts, weights, bias


def _evaluate_regression(feature_matrix: np.ndarray, weights: np.ndarray, bias: float, labels: np.ndarray, metric_labels: Sequence[int]) -> Tuple[float, float, np.ndarray]:
    if feature_matrix.size == 0:
        feature_matrix = np.zeros((len(labels), len(weights)), dtype=int)
    coef = weights
    if coef.size < feature_matrix.shape[1]:
        coef = np.concatenate([coef, np.zeros(feature_matrix.shape[1] - coef.size)])
    elif coef.size > feature_matrix.shape[1]:
        coef = coef[: feature_matrix.shape[1]]
    logits = np.dot(feature_matrix, coef) + bias
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    acc, f1 = compute_accuracy_and_macro_f1(labels, preds, labels=metric_labels)
    return acc, f1, preds


def _build_label_lookup(data_loader: TaskDataLoader, *dfs: pd.DataFrame) -> Dict[int, str]:
    label_mapping = data_loader.label_mapping
    label_lookup: Dict[int, str] = {idx: key for key, idx in label_mapping.items()}
    label_column = data_loader.task.label_name
    for df in dfs:
        if df is None or df.empty or label_column not in df:
            continue
        for value in df[label_column].dropna().unique():
            normalized = _normalize_label(value)
            idx = label_mapping.get(normalized)
            if idx is not None:
                label_lookup[idx] = str(value)
    return label_lookup


def _prepare_datasets(config: Dict, task_root: Optional[str], repeat_index: int):
    config = copy.deepcopy(config)
    if task_root:
        config["data_root"] = task_root
        config["data_roots"] = None
    base_seed = int(config.get("data_seed", 42))
    config["data_seed"] = base_seed + repeat_index

    data_loader = TaskDataLoader(config)
    dataset = data_loader.load_dataset()
    train_df = dataset.train.reset_index(drop=True)
    val_df = dataset.val.reset_index(drop=True)
    test_df = dataset.test.reset_index(drop=True)
    return data_loader, train_df, val_df, test_df


def _score_with_rule_learning(
    rule_texts: List[str],
    rule_learning_cfg: Dict,
    prompt,
    data_loader: TaskDataLoader,
    test_df: pd.DataFrame,
    test_labels: np.ndarray,
    metric_labels: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    checker = RuleScorer(
        rule_learning_cfg,
        prompt,
        data_loader.label_mapping,
        label_values=getattr(data_loader, "label_values", None),
        disable_progress=True,
    )
    rule_vectors = checker.batch_score(rule_texts, test_df, labels=test_labels.tolist())
    feature_matrix = np.array(rule_vectors, dtype=int).T if rule_vectors else np.zeros(
        (len(test_df), len(rule_texts)),
        dtype=int,
    )
    bias_term = float(rule_learning_cfg.get("bias", 0.0))
    weights = np.array(rule_learning_cfg.get("weights"), dtype=float)
    acc, f1, preds = _evaluate_regression(feature_matrix, weights, bias_term, test_labels, metric_labels)
    return feature_matrix, preds, acc, f1


def run_eval_only(
    *,
    config_path: str,
    run_dir: str,
    round_id: Optional[int] = None,
    task_root: Optional[str] = None,
    repeat_index: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    raw_config = load_config(config_path)
    evaluation_modes = parse_evaluation_modes(raw_config)
    LOGGER.info("Evaluation modes: %s", ", ".join(evaluation_modes))

    if repeat_index is None:
        repeat_index = _parse_repeat_index(run_dir, fallback=0)

    if not task_root:
        task_root = raw_config.get("data_root")
        roots = raw_config.get("data_roots") or []
        if not task_root and roots:
            if len(roots) > 1:
                raise ValueError("config.data_roots contains multiple tasks; specify one with --task_root")
            task_root = roots[0]
    if not task_root:
        raise ValueError("Missing task path data_root; set it in the config or via arguments.")

    data_loader, train_df, val_df, test_df = _prepare_datasets(raw_config, task_root, repeat_index)
    label_mapping = data_loader.label_mapping
    metric_labels = sorted(set(label_mapping.values()))
    label_lookup = _build_label_lookup(data_loader, train_df, val_df, test_df)
    test_labels = np.array(map_labels(test_df, label_mapping, data_loader.task.label_name))
    test_label_list = test_labels.tolist()

    # Rules and weights.
    best_round, snapshot_dir, rule_texts, weights, bias = _load_rule_snapshot(run_dir, round_id)
    LOGGER.info("Using rule snapshot: %s (round=%d)", snapshot_dir, best_round)
    active_indices = [idx for idx, w in enumerate(weights) if abs(w) > 1e-12]
    active_texts = [rule_texts[idx] for idx in active_indices]
    weighted_hypotheses = [(rule_texts[idx], float(weights[idx])) for idx in active_indices]

    llm_cfg = raw_config.get("llms", {}) or {}
    generation_cfg = copy.deepcopy(llm_cfg.get("generation") or {})
    evaluation_llm_cfg = copy.deepcopy(llm_cfg.get("evaluation") or {})
    shared_client_cfg = copy.deepcopy(llm_cfg.get("client") or {})

    prompt = BasePrompt(data_loader.task)

    evaluation_model_names = generation_cfg.pop("evaluation_models", None)
    if evaluation_model_names is None:
        evaluation_model_names = []
    elif not isinstance(evaluation_model_names, list):
        raise ValueError("config.llms.generation.evaluation_models must be a list")
    evaluation_model_names = [str(name) for name in evaluation_model_names if str(name).strip()]
    if not evaluation_model_names:
        fallback = evaluation_llm_cfg.get("model")
        if fallback:
            if isinstance(fallback, list):
                evaluation_model_names = [str(name) for name in fallback if str(name).strip()]
            else:
                evaluation_model_names = [str(fallback)]

    evaluation_model_cfgs: List[Tuple[str, Dict]] = []
    for model_name in evaluation_model_names:
        per_model_cfg = prepare_llm_config(
            evaluation_llm_cfg,
            model_name,
            shared_client=shared_client_cfg,
        )
        evaluation_model_cfgs.append((model_name, per_model_cfg))

    feature_mode = resolve_feature_mode(raw_config)
    bert_test_features = None
    if feature_mode == "bert_mlp":
        from rlie.bert.mlp import predict_saved_mlp_features

        mlp_dir = os.path.join(run_dir, "rule_mlps", f"round_{int(best_round):02d}")
        LOGGER.info("Experiment A: using BERT MLP continuous features from %s", mlp_dir)
        bert_test_features = predict_saved_mlp_features(
            config=raw_config,
            task=data_loader.task,
            task_root=task_root,
            frame=test_df,
            mlp_dir=mlp_dir,
            rule_texts=rule_texts,
        )

    # Experiment A: regression features and baseline. bert_mlp uses the saved MLP;
    # otherwise rescore with the first evaluation model.
    regression_results: Dict[str, Dict[str, float]] = {}
    if bert_test_features is not None or evaluation_model_cfgs:
        if bert_test_features is not None:
            model_name = "bert_mlp"
            feature_matrix = bert_test_features
        else:
            model_name, cfg = evaluation_model_cfgs[0]
            LOGGER.info("Experiment A: regression rescore - running only the first evaluation model: %s", model_name)
            checker = RuleScorer(
                cfg,
                prompt,
                data_loader.label_mapping,
                label_values=getattr(data_loader, "label_values", None),
                disable_progress=True,
            )
            rule_vectors = checker.batch_score(rule_texts, test_df, labels=test_label_list)
            feature_matrix = np.array(rule_vectors, dtype=int).T if rule_vectors else np.zeros(
                (len(test_df), len(rule_texts)),
                dtype=int,
            )
        acc_reg, f1_reg, _ = _evaluate_regression(
            feature_matrix,
            weights,
            bias,
            test_labels,
            metric_labels,
        )
        safe_name = sanitize_identifier(model_name)
        save_json_safe(
            os.path.join(run_dir, f"final_test_metrics_regression_only__{safe_name}.json"),
            {
                "accuracy": acc_reg,
                "f1": f1_reg,
                "coverage": 1.0,
                "round": best_round,
                "mode": "regression_only",
            },
        )
        regression_results[model_name] = {
            "regression_only": {
                "accuracy": acc_reg,
                "f1": f1_reg,
                "coverage": 1.0,
            }
        }

    # Experiment B: use the first evaluation model's baseline features and regression
    # predictions, then run the three LLM strategies across every evaluation model.
    if not evaluation_model_cfgs:
        LOGGER.warning("evaluation_models is empty; skipping experiment B")
        return {
            "experiment_A": regression_results,
            "experiment_B": {},
        }

    llm_modes_only = [mode for mode in evaluation_modes if mode != "regression_only"]
    if not llm_modes_only:
        return {
            "experiment_A": regression_results,
            "experiment_B": {},
        }

    # Baseline features and predictions: BERT MLP continuous scores, or a rescore from
    # the first evaluation model (used by the LR and Label strategies).
    if bert_test_features is not None:
        base_model_name = "bert_mlp"
        base_features = bert_test_features
    else:
        base_model_name, base_cfg = evaluation_model_cfgs[0]
        base_checker = RuleScorer(
            base_cfg,
            prompt,
            data_loader.label_mapping,
            label_values=getattr(data_loader, "label_values", None),
            disable_progress=True,
        )
        base_vectors = base_checker.batch_score(rule_texts, test_df, labels=test_label_list)
        base_features = np.array(base_vectors, dtype=int).T if base_vectors else np.zeros(
            (len(test_df), len(rule_texts)),
            dtype=int,
        )
    _, _, base_preds = _evaluate_regression(
        base_features,
        weights,
        bias,
        test_labels,
        metric_labels,
    )

    LOGGER.info(
        "Experiment B: using features from baseline model %s and running LLM strategies across all evaluation models",
        base_model_name,
    )
    llm_results = evaluate_models(
        run_dir=run_dir,
        data_loader=data_loader,
        prompt=prompt,
        evaluation_model_cfgs=evaluation_model_cfgs,
        evaluation_modes=llm_modes_only,
        rule_texts=rule_texts,
        weights=weights,
        bias=bias,
        active_texts=active_texts,
        weighted_hypotheses=weighted_hypotheses,
        # Pure LLM mode does not depend on regression; LR and Label modes need base_preds.
        final_test_preds=base_preds,
        test_df=test_df,
        test_label_list=test_label_list,
        test_labels=test_labels,
        metric_labels=metric_labels,
        label_lookup=label_lookup,
        best_round=best_round,
        skip_regression_model=None,
    )
    # Return experiment A and experiment B separately.
    return {
        "experiment_A": regression_results,
        "experiment_B": llm_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run only the evaluation stage from an existing rule snapshot")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the config file")
    parser.add_argument("--run_dir", required=True, help="Output directory of a finished run, e.g. outputs/run_.../<task>_repeat1")
    parser.add_argument("--round", type=int, default=None, help="Rule round to use (defaults to the round in final_round_metrics)")
    parser.add_argument("--task_root", type=str, default=None, help="data_root to use (e.g. real/deceptive_reviews)")
    parser.add_argument("--repeat_index", type=int, default=None, help="Zero-based repeat index; if omitted, inferred from a _repeatN directory name")
    args = parser.parse_args()

    results = run_eval_only(
        config_path=args.config,
        run_dir=args.run_dir,
        round_id=args.round,
        task_root=args.task_root,
        repeat_index=args.repeat_index,
    )
    LOGGER.info("Evaluation finished; results are in %s", args.run_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
