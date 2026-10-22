"""CPU tests for joint BERT fine-tuning and per-rule loss balancing."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rlie.bert.dynamic import DynamicRuleModel, dynamic_settings
from rlie.bert.weighting import AdaptiveLossWeighter, last_feedforward_parameter
from rlie.core.feature_manager import RuleFeatureStore


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("torch is not installed") from exc
    return torch


class TestLossWeights(unittest.TestCase):
    def test_adaptive_weights_start_uniform_and_track_loss_share(self):
        torch = _require_torch()
        weighter = AdaptiveLossWeighter(2)
        self.assertTrue(np.allclose(weighter.weights(), [0.5, 0.5]))

        first = torch.tensor(1.0, requires_grad=True)
        second = torch.tensor(3.0, requires_grad=True)
        weighter.backward([first, second])
        self.assertAlmostEqual(float(first.grad), 0.5)
        self.assertAlmostEqual(float(second.grad), 0.5)
        self.assertTrue(np.allclose(weighter.weights(), [0.25, 0.75]))

        next_first = torch.tensor(1.0, requires_grad=True)
        next_second = torch.tensor(3.0, requires_grad=True)
        weighter.backward([next_first, next_second])
        self.assertAlmostEqual(float(next_first.grad), 0.25)
        self.assertAlmostEqual(float(next_second.grad), 0.75)

    def test_dynamic_learning_rates_are_separate(self):
        settings = dynamic_settings(
            {
                "bert": {
                    "mlp": {"learning_rate": 0.2},
                    "dynamic": {
                        "strategy": "grad_norm",
                        "bert_learning_rate": 1e-5,
                        "mlp_learning_rate": 3e-4,
                    },
                }
            }
        )
        self.assertEqual(settings["strategy"], "gradnorm")
        self.assertEqual(settings["bert_learning_rate"], 1e-5)
        self.assertEqual(settings["mlp_learning_rate"], 3e-4)
        with self.assertRaises(ValueError):
            dynamic_settings({"bert": {"dynamic": {"strategy": "unknown"}}})

    def test_last_feedforward_parameter(self):
        torch = _require_torch()
        from torch import nn

        class BertLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.output = nn.Module()
                self.output.dense = nn.Linear(8, 4)

        class BertEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.ModuleList([BertLayer(), BertLayer()])

        class Bert(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = BertEncoder()

        bert = Bert()
        self.assertIs(last_feedforward_parameter(bert), bert.encoder.layer[-1].output.dense.weight)

        class ModernLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.Wi = nn.Linear(4, 8)
                self.mlp.Wo = nn.Linear(8, 4)

        class ModernBert(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([ModernLayer(), ModernLayer()])

        modern = ModernBert()
        self.assertIs(last_feedforward_parameter(modern), modern.layers[-1].mlp.Wo.weight)
        self.assertIsInstance(last_feedforward_parameter(modern), torch.nn.Parameter)


def _tiny_bert(torch):
    from torch import nn

    class Block(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.Wi = nn.Linear(hidden, hidden * 2)
            self.mlp.Wo = nn.Linear(hidden * 2, hidden)

    class TinyBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.layers = nn.ModuleList([Block(8)])
            self.hidden_size = 8

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embed(input_ids)
            hidden = self.layers[0].mlp.Wo(torch.relu(self.layers[0].mlp.Wi(hidden)))
            return SimpleNamespace(last_hidden_state=hidden)

    return TinyBert()


def _tokens(torch, rows):
    return {
        "input_ids": torch.randint(0, 16, (rows, 5)),
        "attention_mask": torch.ones(rows, 5, dtype=torch.long),
    }


def _store(train_rows, val_rows, rules):
    train_df = pd.DataFrame({"label": np.zeros(train_rows, dtype=int)})
    val_df = pd.DataFrame({"label": np.zeros(val_rows, dtype=int)})
    store = RuleFeatureStore(samples=train_df, label_column="label", val_samples=val_df)
    generator = np.random.default_rng(0)
    train_vectors = [generator.choice([-1, 0, 1], size=train_rows).tolist() for _ in rules]
    val_vectors = [generator.choice([-1, 0, 1], size=val_rows).tolist() for _ in rules]
    store.add_rules(list(rules), train_vectors, 0, val_vectors=val_vectors)
    return store


class TestDynamicFineTune(unittest.TestCase):
    def test_new_rules_keep_bert_and_existing_mlp_weights(self):
        torch = _require_torch()
        settings = dynamic_settings(
            {
                "bert": {
                    "mlp": {"hidden_dims": [8], "dropout": 0.0},
                    "dynamic": {
                        "strategy": "adaptive_weighting",
                        "bert_learning_rate": 1e-3,
                        "mlp_learning_rate": 1e-2,
                        "max_epochs": 1,
                        "patience": 1,
                        "batch_size": 8,
                    },
                }
            }
        )
        bank = DynamicRuleModel(
            _tiny_bert(torch),
            _tokens(torch, 12),
            _tokens(torch, 6),
            torch.device("cpu"),
            settings,
            seed=0,
        )
        store = _store(12, 6, ["first rule"])
        train_features, val_features = bank.sync_and_predict(store)
        self.assertEqual(train_features.shape, (12, 1))
        self.assertEqual(val_features.shape, (6, 1))
        self.assertTrue(np.all(train_features >= -1.0) and np.all(train_features <= 1.0))

        bert_weight = bank.encoder.layers[0].mlp.Wo.weight
        first_head = bank._models[id(store.rules[0])]
        first_weight = next(first_head.parameters())
        with torch.no_grad():
            bert_weight.fill_(0.123)
            first_weight.fill_(0.456)
        sentinel_bert = bert_weight.detach().clone()
        sentinel_mlp = first_weight.detach().clone()
        first_rule = store.rules[0]
        bert_id = id(bank.encoder)
        head_id = id(first_head)

        captured = {}
        original_fit = bank._fit

        def capture_then_fit(feature_store, rules):
            captured["bert"] = bank.encoder.layers[0].mlp.Wo.weight.detach().clone()
            captured["mlp"] = next(bank._models[id(first_rule)].parameters()).detach().clone()
            captured["same_head"] = bank._models[id(first_rule)] is first_head
            return original_fit(feature_store, rules)

        bank._fit = capture_then_fit
        store.add_rules(
            ["second rule"],
            [np.zeros(12, dtype=int).tolist()],
            1,
            val_vectors=[np.zeros(6, dtype=int).tolist()],
        )
        again, _ = bank.sync_and_predict(store)
        self.assertEqual(again.shape, (12, 2))
        self.assertEqual(id(bank.encoder), bert_id)
        self.assertTrue(captured["same_head"])
        self.assertEqual(id(bank._models[id(first_rule)]), head_id)
        self.assertTrue(torch.allclose(captured["bert"], sentinel_bert))
        self.assertTrue(torch.allclose(captured["mlp"], sentinel_mlp))
        self.assertIn(id(store.rules[1]), bank._models)

    def test_classification_features_are_discrete_labels(self):
        torch = _require_torch()
        settings = dynamic_settings(
            {
                "bert": {
                    "mlp": {"head": "classification", "hidden_dims": [8], "dropout": 0.0},
                    "dynamic": {
                        "strategy": "adaptive_weighting",
                        "bert_learning_rate": 1e-3,
                        "mlp_learning_rate": 1e-2,
                        "max_epochs": 1,
                        "patience": 1,
                        "batch_size": 8,
                    },
                }
            }
        )
        self.assertEqual(settings["head"], "classification")
        self.assertEqual(settings["metric"], "accuracy")
        bank = DynamicRuleModel(
            _tiny_bert(torch),
            _tokens(torch, 12),
            _tokens(torch, 6),
            torch.device("cpu"),
            settings,
            seed=0,
        )
        train_features, _ = bank.sync_and_predict(_store(12, 6, ["first rule", "second rule"]))
        self.assertEqual(train_features.shape, (12, 2))
        self.assertTrue(set(np.unique(train_features)).issubset({-1.0, 0.0, 1.0}))
        logits = next(iter(bank._models.values()))(torch.zeros(4, bank.hidden_size))
        self.assertEqual(tuple(logits.shape), (4, 3))

    def test_gradnorm_step_on_shared_feedforward_weight(self):
        torch = _require_torch()
        try:
            from gradnorm_pytorch import GradNormLossWeighter as _  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("gradnorm-pytorch is not installed") from exc
        from torch import nn

        from rlie.bert.weighting import build_loss_weighter

        shared = nn.Linear(4, 4)
        head_a = nn.Linear(4, 1)
        head_b = nn.Linear(4, 1)
        features = torch.randn(8, 4)
        target_a = torch.randn(8)
        target_b = torch.randn(8)
        hidden = torch.tanh(shared(features))
        losses = [
            nn.functional.mse_loss(head_a(hidden).squeeze(-1), target_a),
            nn.functional.mse_loss(head_b(hidden).squeeze(-1), target_b),
        ]
        weighter = build_loss_weighter(
            "gradnorm",
            2,
            shared.weight,
            {"gradnorm_learning_rate": 1e-3, "gradnorm_alpha": 0.0},
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": shared.parameters(), "lr": 1e-3},
                {"params": list(head_a.parameters()) + list(head_b.parameters()), "lr": 1e-2},
            ]
        )
        optimizer.zero_grad(set_to_none=True)
        weighter.backward(losses)
        optimizer.step()
        self.assertTrue(all(np.isfinite(weighter.weights())))
        self.assertTrue(torch.isfinite(shared.weight).all())


if __name__ == "__main__":
    unittest.main()
