"""Config splitting and retained-hypothesis BERT training."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rlie.core.reporting import load_run_samples, save_run_samples
from rlie.datasets.data_loader import apply_external_configs, load_config
from rlie.train_retained import (
    build_retained_store,
    fit_retained,
    load_retained_store,
    read_round_feature_csv,
)


REPO = Path(__file__).resolve().parents[1]


class _StubBank:
    def __init__(self):
        self.saved = None

    def sync_and_predict(self, store):
        train = store.to_sample_feature_matrix().astype(np.float32)
        val = None
        if store.val_samples is not None:
            val = store.to_sample_feature_matrix("val").astype(np.float32)
        return train, val

    def save_round(self, round_idx, output_dir, rules):
        directory = os.path.join(output_dir, "rule_mlps", f"round_{int(round_idx):02d}")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump({"rules": [rule.text for rule in rules]}, handle)
        self.saved = directory
        return directory


def _write_run(root: str) -> None:
    os.makedirs(os.path.join(root, "rule_snapshots"), exist_ok=True)
    train = pd.DataFrame(
        {
            "text": [f"row {idx}" for idx in range(8)],
            "label": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        }
    )
    val = pd.DataFrame({"text": ["v0", "v1"], "label": ["yes", "no"]})
    save_run_samples(root, train, val, val.iloc[0:0])
    with open(os.path.join(root, "final_round_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"round": 2, "feature_mode": "llm"}, handle)
    rules = {
        "round": 2,
        "bias": 0.1,
        "rules": [
            {"rule_index": 0, "rule": "unused", "weight": 0.0, "source_round": 1},
            {
                "rule_index": 1,
                "rule": "kept",
                "weight": 0.4,
                "source_round": 2,
                "train_accuracy": 0.75,
                "val_accuracy": 0.5,
            },
        ],
    }
    with open(os.path.join(root, "rule_snapshots", "round_02_rules.json"), "w", encoding="utf-8") as handle:
        json.dump(rules, handle)
    train_rules = np.array(
        [
            [1, 0],
            [-1, 1],
            [1, 0],
            [-1, 1],
            [1, -1],
            [-1, 0],
            [1, 1],
            [-1, -1],
        ],
        dtype=int,
    )
    train_csv = pd.DataFrame(train_rules, columns=["rule_0", "rule_1"])
    train_csv.insert(0, "label", [1, 0, 1, 0, 1, 0, 1, 0])
    train_csv.to_csv(os.path.join(root, "round_02_train_dataset.csv"), index=False)
    val_csv = pd.DataFrame([[1, 0], [-1, 1]], columns=["rule_0", "rule_1"])
    val_csv.insert(0, "label", [1, 0])
    val_csv.to_csv(os.path.join(root, "round_02_val_dataset.csv"), index=False)


class TestSplitConfigs(unittest.TestCase):
    def test_default_config_loads_the_three_files(self):
        previous = os.getcwd()
        os.chdir(REPO)
        try:
            config = load_config(str(REPO / "configs" / "default.yaml"))
        finally:
            os.chdir(previous)
        self.assertEqual(config["bert"]["model_name"], "google-bert/bert-base-cased")
        self.assertEqual(config["bert"]["dynamic"]["strategy"], "gradnorm")
        self.assertEqual(config["bert"]["mlp"]["head"], "classification")
        self.assertNotIn("metric", config["bert"]["mlp"])
        self.assertEqual(config["elastic_net"]["max_iter"], 20000)
        self.assertNotIn("C_grid", config["elastic_net"])

    def test_paths_beside_the_config_file_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "bert.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump({"model_name": "tiny", "max_length": 8}, handle)
            with open(os.path.join(tmp, "head.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump({"head": "regression", "hidden_dims": [4]}, handle)
            with open(os.path.join(tmp, "elastic_net.yaml"), "w", encoding="utf-8") as handle:
                yaml.safe_dump({"max_iter": 10, "cv_folds": 2}, handle)
            config_path = os.path.join(tmp, "experiment.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    {
                        "bert_config": "bert.yaml",
                        "head_config": "head.yaml",
                        "sklearn_config": "elastic_net.yaml",
                    },
                    handle,
                )
            config = load_config(config_path)
        self.assertEqual(config["bert"]["model_name"], "tiny")
        self.assertEqual(config["bert"]["mlp"]["hidden_dims"], [4])
        self.assertEqual(config["elastic_net"]["max_iter"], 10)

    def test_missing_reference_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "experiment.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump({"sklearn_config": "missing.yaml"}, handle)
            with self.assertRaises(FileNotFoundError):
                load_config(config_path)


class TestRetainedRules(unittest.TestCase):
    def test_nonzero_rules_keep_their_silver_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(tmp)
            labels, matrix = read_round_feature_csv(os.path.join(tmp, "round_02_train_dataset.csv"))
            self.assertEqual(matrix.shape, (8, 2))
            self.assertTrue(np.array_equal(labels, [1, 0, 1, 0, 1, 0, 1, 0]))
            store, train_df, val_df, round_id = load_retained_store(
                tmp,
                {},
                task_root="real/headline_binary",
                repeat_index=0,
                label_mapping={"yes": 1, "no": 0},
                label_column="label",
            )
            reloaded_train, reloaded_val, _test = load_run_samples(tmp)
        self.assertEqual(round_id, 2)
        self.assertEqual([rule.text for rule in store.rules], ["kept"])
        self.assertEqual(store.features.shape, (1, len(train_df)))
        self.assertTrue(np.array_equal(store.features[0], [0, 1, 0, 1, -1, 0, 1, -1]))
        self.assertTrue(np.array_equal(store.val_features[0], [0, 1]))
        self.assertEqual(len(val_df), 2)
        self.assertEqual(reloaded_train["text"].tolist(), train_df["text"].tolist())
        self.assertEqual(reloaded_val["label"].tolist(), ["yes", "no"])

    def test_bert_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(tmp, exist_ok=True)
            with open(os.path.join(tmp, "final_round_metrics.json"), "w", encoding="utf-8") as handle:
                json.dump({"round": 0, "feature_mode": "bert_static"}, handle)
            with self.assertRaises(ValueError):
                load_retained_store(
                    tmp,
                    {},
                    task_root="real/headline_binary",
                    repeat_index=0,
                    label_mapping={"yes": 1},
                    label_column="label",
                )

    def test_refit_uses_only_the_retained_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_run(tmp)
            store, train_df, val_df, round_id = load_retained_store(
                tmp,
                {},
                task_root="real/headline_binary",
                repeat_index=0,
                label_mapping={"yes": 1, "no": 0},
                label_column="label",
            )
            bank = _StubBank()
            output = os.path.join(tmp, "retained")
            payload = fit_retained(
                {
                    "model_selection_metric": "accuracy",
                    "elastic_net": {
                        "C_grid": [1.0],
                        "l1_ratio_grid": [0.5],
                        "cv_folds": 2,
                        "max_iter": 200,
                        "n_jobs": 1,
                    },
                },
                task=None,
                task_root="real/headline_binary",
                train_df=train_df,
                val_df=val_df,
                store=store,
                train_labels=np.array([1, 0, 1, 0, 1, 0, 1, 0]),
                val_labels=np.array([1, 0]),
                output_dir=output,
                feature_mode="bert_static",
                source_round=round_id,
                source_run_dir=tmp,
                bank=bank,
            )
            self.assertTrue(os.path.exists(os.path.join(bank.saved, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(output, "rule_snapshots", "round_02_rules.json")))
        self.assertEqual(payload["rules"], ["kept"])
        self.assertEqual(payload["feature_mode"], "bert_static")
        self.assertIn("val", payload)

    def test_all_zero_weights_raise(self):
        train = pd.DataFrame({"label": [1, 0]})
        with self.assertRaises(ValueError):
            build_retained_store(
                train,
                None,
                [{"rule": "nothing", "weight": 0.0, "rule_index": 0}],
                np.array([[1], [-1]]),
                None,
                label_column="label",
                source_round=0,
            )


class TestExternalConfigHelper(unittest.TestCase):
    def test_head_config_overrides_an_inline_mlp(self):
        with tempfile.TemporaryDirectory() as tmp:
            head_path = os.path.join(tmp, "head.yaml")
            with open(head_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump({"head": "classification"}, handle)
            merged = apply_external_configs(
                {"bert": {"model_name": "inline", "mlp": {"head": "regression"}}, "head_config": head_path},
            )
        self.assertEqual(merged["bert"]["model_name"], "inline")
        self.assertEqual(merged["bert"]["mlp"]["head"], "classification")


if __name__ == "__main__":
    unittest.main()
