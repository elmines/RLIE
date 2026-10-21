"""Tests for BERT input templates and the per-rule MLP feature mode."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rlie.bert.embed import pool_cls, tokenizer_arguments
from rlie.bert.mlp import _improved, build_rule_mlp, load_rule_mlps, resolve_device
from rlie.bert.pipeline import build_mlp_bank, regression_feature_matrices, resolve_feature_mode
from rlie.bert.store import load_embedding_index, meta_payload, save_split, write_meta
from rlie.bert.templates import TASK_BERT_BUILDERS, headline_binary_segments, sample_key, segments_for_frame
from rlie.core.feature_manager import RuleFeatureStore
from rlie.core.modeling import SoftmaxTrainer


ROOT = Path(__file__).resolve().parents[1]


class TaskStub:
    def __init__(self, task_name, fields):
        self.task_name = task_name
        self.bert_input_fields = list(fields)


class TestTemplates(unittest.TestCase):
    def test_registered_templates_match_task_yaml(self):
        specs = list((ROOT / "rlie" / "task_specs").rglob("task.yaml"))
        self.assertGreaterEqual(len(specs), 6)
        for path in specs:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            builder = TASK_BERT_BUILDERS[data["task_name"]]
            self.assertEqual(tuple(data["bert_input"]["fields"]), tuple(builder.fields))

    def test_headline_pair_is_cls_sep_segments(self):
        row = pd.Series({"headline_1": "Alpha", "headline_2": "Beta"})
        segments = headline_binary_segments(row)
        self.assertEqual(segments, ["Alpha", "Beta"])
        text, text_pair = tokenizer_arguments([segments])
        self.assertEqual(text, ["Alpha"])
        self.assertEqual(text_pair, ["Beta"])
        single, pair = tokenizer_arguments([["only text"]])
        self.assertEqual(single, ["only text"])
        self.assertIsNone(pair)

    def test_sample_key_depends_on_segment_order(self):
        self.assertEqual(sample_key(["Alpha", "Beta"]), sample_key(["Alpha", "Beta"]))
        self.assertNotEqual(sample_key(["Alpha", "Beta"]), sample_key(["Beta", "Alpha"]))


class TestEmbeddingIndex(unittest.TestCase):
    def test_align_follows_dataframe_order(self):
        task = TaskStub("headline_binary", ["headline_1", "headline_2"])
        frame = pd.DataFrame(
            {
                "headline_1": ["h1", "h2"],
                "headline_2": ["a", "b"],
            }
        )
        segments = segments_for_frame(task, frame)
        keys = [sample_key(segment) for segment in segments]
        embeddings = np.arange(8, dtype=np.float32).reshape(2, 4)
        with tempfile.TemporaryDirectory() as tmp:
            save_split(os.path.join(tmp, "train.npz"), embeddings, keys)
            write_meta(
                tmp,
                meta_payload(
                    task=task,
                    task_root="real/headline_binary",
                    model_name="bert-base-uncased",
                    max_length=32,
                    hidden_size=4,
                    splits={"train": 2},
                ),
            )
            index = load_embedding_index(tmp)
            reordered = frame.iloc[[1, 0]].reset_index(drop=True)
            aligned = index.align(task, reordered)
            self.assertTrue(np.allclose(aligned[0], embeddings[1]))
            self.assertTrue(np.allclose(aligned[1], embeddings[0]))
            missing = pd.DataFrame({"headline_1": ["missing"], "headline_2": ["row"]})
            with self.assertRaises(KeyError):
                index.align(task, missing)


class TestFeatureMode(unittest.TestCase):
    def test_default_mode_keeps_ternary_features(self):
        self.assertEqual(resolve_feature_mode({}), "llm")
        self.assertEqual(resolve_feature_mode({"feature_mode": "llm"}), "llm")
        self.assertEqual(resolve_feature_mode({"feature_mode": "bert_static"}), "bert_static")
        self.assertEqual(resolve_feature_mode({"feature_mode": "bert_mlp"}), "bert_static")
        self.assertEqual(resolve_feature_mode({"feature_mode": "bert_dynamic"}), "bert_dynamic")
        with self.assertRaises(ValueError):
            resolve_feature_mode({"feature_mode": "something-else"})
        self.assertIsNone(build_mlp_bank({}, task=None, task_root="", train_df=None, val_df=None))

        class Store:
            def __init__(self):
                self.features = np.array([[1, -1, 0], [0, 1, -1]], dtype=np.int8)
                self.val_features = np.array([[1, 0], [-1, 1]], dtype=np.int8)
                self.val_samples = object()

            def to_sample_feature_matrix(self, split="train"):
                if split == "train":
                    return self.features.T
                return self.val_features.T

        store = Store()
        train, val = regression_feature_matrices(store, None)
        self.assertTrue(np.array_equal(train, store.features.T))
        self.assertTrue(np.array_equal(val, store.val_features.T))


class TestRuleMLP(unittest.TestCase):
    def test_predictions_stay_continuous_in_range(self):
        torch = _require_torch()
        model = build_rule_mlp(4, [], 0.0)
        from torch import nn

        linear = [module for module in model.modules() if isinstance(module, nn.Linear)][-1]
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.fill_(0.5)
        from rlie.bert.mlp import TrainedRuleMLP

        predictions = TrainedRuleMLP(model, torch.device("cpu"), {}).predict(np.zeros((5, 4), dtype=np.float32))
        self.assertTrue(np.allclose(predictions, np.tanh(0.5), atol=1e-5))
        self.assertTrue(np.all((predictions > -1.0) & (predictions < 1.0)))
        self.assertFalse(np.all(np.isin(np.round(predictions, 5), [-1.0, 0.0, 1.0])))

    def test_pool_cls_uses_position_zero(self):
        torch = _require_torch()
        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        pooled = pool_cls(hidden)
        self.assertTrue(torch.equal(pooled, hidden[:, 0, :]))

    def test_device_resolution(self):
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")
        with self.assertRaises(RuntimeError):
            resolve_device("cuda:99")

    def test_convergence_rule(self):
        self.assertTrue(_improved(0.4, None, "min", 1e-4))
        self.assertTrue(_improved(0.2, 0.4, "min", 1e-4))
        self.assertFalse(_improved(0.4 - 1e-6, 0.4, "min", 1e-4))
        self.assertTrue(_improved(0.8, 0.5, "max", 1e-4))
        self.assertFalse(_improved(float("nan"), 0.5, "min", 1e-4))

    def test_mlp_fits_silver_labels_and_reloads(self):
        _require_torch()
        from rlie.bert.mlp import RuleMLPBank

        rng = np.random.default_rng(0)
        train_x = rng.normal(size=(48, 6)).astype(np.float32)
        val_x = rng.normal(size=(16, 6)).astype(np.float32)
        train_y = np.sign(train_x[:, 0]).astype(np.int8)
        val_y = np.sign(val_x[:, 0]).astype(np.int8)
        train_df = pd.DataFrame({"label": np.zeros(len(train_x), dtype=int)})
        val_df = pd.DataFrame({"label": np.zeros(len(val_x), dtype=int)})
        store = RuleFeatureStore(samples=train_df, label_column="label", val_samples=val_df)
        store.add_rules(
            ["uses the first direction", "uses the opposite direction"],
            [train_y.tolist(), (-train_y).tolist()],
            0,
            val_vectors=[val_y.tolist(), (-val_y).tolist()],
        )
        bank = RuleMLPBank(
            train_x,
            val_x,
            resolve_device("cpu"),
            {
                "hidden_dims": [16],
                "dropout": 0.0,
                "learning_rate": 0.05,
                "weight_decay": 0.0,
                "batch_size": 16,
                "max_epochs": 25,
                "patience": 4,
                "min_delta": 1e-4,
                "metric": "mse",
            },
            seed=7,
        )
        import rlie.bert.mlp as mlp_mod

        calls = {"n": 0}
        original = mlp_mod.train_rule_mlp

        def counted(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        mlp_mod.train_rule_mlp = counted
        try:
            train_features, val_features = bank.sync_and_predict(store)
            self.assertEqual(calls["n"], 2)
            again_train, again_val = bank.sync_and_predict(store)
            self.assertEqual(calls["n"], 2)
        finally:
            mlp_mod.train_rule_mlp = original

        self.assertEqual(train_features.shape, (len(train_x), 2))
        self.assertEqual(val_features.shape, (len(val_x), 2))
        self.assertTrue(np.all(train_features >= -1.0) and np.all(train_features <= 1.0))
        self.assertTrue(np.allclose(train_features, again_train))
        self.assertTrue(np.allclose(val_features, again_val))
        zero_mse = float(np.mean(train_y.astype(np.float32) ** 2))
        fit_mse = float(np.mean((train_features[:, 0] - train_y) ** 2))
        self.assertLess(fit_mse, zero_mse * 0.5)
        # Silver labels stay ternary for rule metrics; only the regression features are continuous.
        self.assertTrue(np.array_equal(store.features[0], train_y))

        labels = (train_x[:, 0] > 0).astype(int)
        trainer = SoftmaxTrainer(
            {
                "model_selection_metric": "accuracy",
                "elastic_net": {
                    "C_grid": [1.0],
                    "l1_ratio_grid": [0.5],
                    "cv_folds": 2,
                    "max_iter": 200,
                    "n_jobs": 1,
                },
            }
        )
        fitted = trainer.train(train_features, labels)
        self.assertEqual(fitted.probabilities.shape, (len(labels),))

        with tempfile.TemporaryDirectory() as tmp:
            saved = bank.save_round(3, tmp, store.rules)
            loaded = load_rule_mlps(
                saved,
                resolve_device("cpu"),
                expected_rules=[rule.text for rule in store.rules],
            )
            reloaded = np.stack([model.predict(train_x) for model in loaded], axis=1)
            self.assertTrue(np.allclose(reloaded, train_features, atol=1e-5))
            with self.assertRaises(ValueError):
                load_rule_mlps(saved, resolve_device("cpu"), expected_rules=["other rule"])


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("torch is not installed") from exc
    return torch


if __name__ == "__main__":
    unittest.main()
