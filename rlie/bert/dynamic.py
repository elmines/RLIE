"""Fine-tune one shared BERT encoder together with one MLP per rule.

Later rounds keep the encoder weights and every MLP that is still in the rule
set. Only rules that appear for the first time get a new MLP. Each rule has its
own loss term, combined by the configured weighting strategy.
"""

from __future__ import annotations

import json
from rlie.utils.logger_config import LoggerConfig
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from rlie.bert.embed import pool_cls, tokenizer_arguments
from rlie.bert.mlp import (
    _METRIC_DIRECTION,
    _TRAIN_LOCK,
    _copy_state,
    _improved,
    _metric_value,
    _torch_load,
    build_rule_mlp,
    load_rule_mlps,
    mlp_settings,
    resolve_device,
)
from rlie.bert.templates import segments_for_frame
from rlie.bert.weighting import build_loss_weighter, last_feedforward_parameter


LOGGER = LoggerConfig.get_logger("rlie.bert.dynamic")


def dynamic_settings(config: dict) -> dict:
    bert_cfg = config.get("bert") or {}
    dynamic_cfg = dict(bert_cfg.get("dynamic") or {})
    mlp_cfg = mlp_settings(config)
    strategy = str(dynamic_cfg.get("strategy", "adaptive_weighting")).strip().lower().replace("-", "_")
    aliases = {
        "adaptive": "adaptive_weighting",
        "adaptive_weighting": "adaptive_weighting",
        "gradnorm": "gradnorm",
        "grad_norm": "gradnorm",
    }
    if strategy not in aliases:
        raise ValueError("bert.dynamic.strategy must be 'adaptive_weighting' or 'gradnorm'")
    return {
        "strategy": aliases[strategy],
        "bert_learning_rate": float(dynamic_cfg.get("bert_learning_rate", 2e-5)),
        "mlp_learning_rate": float(dynamic_cfg.get("mlp_learning_rate", mlp_cfg["learning_rate"])),
        "weight_decay": float(dynamic_cfg.get("weight_decay", mlp_cfg["weight_decay"])),
        "gradnorm_learning_rate": float(dynamic_cfg.get("gradnorm_learning_rate", 1e-4)),
        "gradnorm_alpha": float(dynamic_cfg.get("gradnorm_alpha", 0.0)),
        "batch_size": max(1, int(dynamic_cfg.get("batch_size", mlp_cfg["batch_size"]))),
        "max_epochs": max(1, int(dynamic_cfg.get("max_epochs", mlp_cfg["max_epochs"]))),
        "patience": max(1, int(dynamic_cfg.get("patience", mlp_cfg["patience"]))),
        "min_delta": float(dynamic_cfg.get("min_delta", mlp_cfg["min_delta"])),
        "metric": mlp_cfg["metric"],
        "hidden_dims": list(mlp_cfg["hidden_dims"]),
        "dropout": float(mlp_cfg["dropout"]),
        "max_length": int(dynamic_cfg.get("max_length", bert_cfg.get("max_length", 512))),
        "model_name": str(dynamic_cfg.get("model_name") or bert_cfg.get("model_name") or "bert-base-uncased"),
        "device": str(dynamic_cfg.get("device") or bert_cfg.get("mlp_device") or bert_cfg.get("device") or "cpu"),
    }


def hidden_size_of(encoder) -> int:
    config = getattr(encoder, "config", None)
    size = getattr(config, "hidden_size", None) if config is not None else None
    if not size:
        size = getattr(encoder, "hidden_size", None)
    if not size:
        raise ValueError("BERT encoder does not expose hidden_size")
    return int(size)


def tokenize_segments(tokenizer, segments: Sequence[Sequence[str]], max_length: int) -> dict:
    import torch

    if not segments:
        empty = torch.zeros((0, 1), dtype=torch.long)
        return {"input_ids": empty, "attention_mask": empty.clone()}
    text, text_pair = tokenizer_arguments(segments, sep_token=tokenizer.sep_token or "[SEP]")
    kwargs = {
        "truncation": True,
        "padding": True,
        "max_length": int(max_length),
        "return_tensors": "pt",
    }
    if text_pair is None:
        encoded = tokenizer(text, **kwargs)
    else:
        encoded = tokenizer(text, text_pair, **kwargs)
    return {
        "input_ids": encoded["input_ids"].cpu(),
        "attention_mask": encoded["attention_mask"].cpu(),
    }


def _forward_cls(encoder, batch: dict):
    output = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    return pool_cls(hidden)


def _slice_tokens(tokens: dict, index, device) -> dict:
    return {key: value[index].to(device) for key, value in tokens.items()}


class DynamicRuleModel:
    """Shared BERT encoder plus one persistent MLP head per rule."""

    def __init__(
        self,
        encoder,
        train_tokens: dict,
        val_tokens: Optional[dict],
        device,
        settings: dict,
        *,
        seed: int,
    ):
        self.encoder = encoder.to(device)
        self.train_tokens = train_tokens
        self.val_tokens = val_tokens
        self.device = device
        self.settings = settings
        self.seed = int(seed)
        self.hidden_size = hidden_size_of(encoder)
        self._models: Dict[int, object] = {}
        self._head_counter = 0
        self._has_fit = False
        self._best_metrics: Dict[int, float] = {}

    @classmethod
    def from_config(cls, config: dict, task, task_root: str, train_df, val_df) -> "DynamicRuleModel":
        from transformers import AutoModel, AutoTokenizer

        if not task_root:
            raise ValueError("feature_mode bert_dynamic requires a task id such as real/headline_binary")
        settings = dynamic_settings(config)
        device = resolve_device(settings["device"])
        LOGGER.info(
            "Loading %s for joint fine-tuning on %s (strategy=%s, bert_lr=%g, mlp_lr=%g)",
            settings["model_name"],
            device,
            settings["strategy"],
            settings["bert_learning_rate"],
            settings["mlp_learning_rate"],
        )
        tokenizer = AutoTokenizer.from_pretrained(settings["model_name"])
        encoder = AutoModel.from_pretrained(settings["model_name"])
        train_tokens = tokenize_segments(
            tokenizer,
            segments_for_frame(task, train_df),
            settings["max_length"],
        )
        val_tokens = None
        if val_df is not None and len(val_df) > 0:
            val_tokens = tokenize_segments(
                tokenizer,
                segments_for_frame(task, val_df),
                settings["max_length"],
            )
        return cls(
            encoder,
            train_tokens,
            val_tokens,
            device,
            settings,
            seed=int(config.get("data_seed", 0)),
        )

    def _sync_heads(self, rules: Sequence) -> bool:
        import torch

        created = False
        alive = {id(rule) for rule in rules}
        for stale in list(self._models):
            if stale not in alive:
                del self._models[stale]
                self._best_metrics.pop(stale, None)
        for rule in rules:
            if id(rule) in self._models:
                continue
            torch.manual_seed(self.seed + self._head_counter)
            self._head_counter += 1
            module = build_rule_mlp(
                self.hidden_size,
                self.settings["hidden_dims"],
                self.settings["dropout"],
            ).to(self.device)
            self._models[id(rule)] = module
            created = True
            LOGGER.info("Initialized a new rule MLP for '%s'", str(rule.text)[:80])
        return created

    def sync_and_predict(self, feature_store):
        with _TRAIN_LOCK:
            rules = list(feature_store.rules)
            created = self._sync_heads(rules)
            if created or not self._has_fit:
                self._fit(feature_store, rules)
                self._has_fit = True
            return self._predict_matrix(self.train_tokens, rules), (
                None if self.val_tokens is None else self._predict_matrix(self.val_tokens, rules)
            )

    def _fit(self, feature_store, rules: Sequence) -> None:
        import torch
        from torch import nn

        silver_train = np.asarray(feature_store.features, dtype=np.float32)
        if silver_train.shape != (len(rules), self.train_tokens["input_ids"].shape[0]):
            raise ValueError(
                "Train silver labels do not align with the tokenized training samples "
                f"({tuple(silver_train.shape)} vs {len(rules)} rules x {self.train_tokens['input_ids'].shape[0]} samples)"
            )
        silver_val = None
        if self.val_tokens is not None:
            silver_val = np.asarray(feature_store.val_features, dtype=np.float32)
            expected = (len(rules), self.val_tokens["input_ids"].shape[0])
            if silver_val.shape != expected:
                raise ValueError(
                    f"Validation silver labels have shape {tuple(silver_val.shape)}; expected {expected}"
                )

        settings = self.settings
        heads = [self._models[id(rule)] for rule in rules]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [param for param in self.encoder.parameters() if param.requires_grad],
                    "lr": settings["bert_learning_rate"],
                },
                {
                    "params": [param for head in heads for param in head.parameters()],
                    "lr": settings["mlp_learning_rate"],
                },
            ],
            weight_decay=settings["weight_decay"],
        )
        grad_parameter = None
        if settings["strategy"] == "gradnorm" and len(rules) < 2:
            LOGGER.info("GradNorm needs at least two rules; using the single rule loss directly")
        elif settings["strategy"] == "gradnorm":
            grad_parameter = last_feedforward_parameter(self.encoder)
        weighter = build_loss_weighter(
            settings["strategy"],
            len(rules),
            grad_parameter,
            settings,
        )
        loss_fn = nn.MSELoss()
        metric_name = settings["metric"]
        direction = _METRIC_DIRECTION[metric_name]
        use_val = silver_val is not None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self._head_counter)
        train_targets = torch.from_numpy(silver_train)
        best_encoder = None
        best_heads = None
        best_metric = None
        best_epoch = -1
        best_rule_metrics: Dict[int, float] = {}
        wait = 0
        epochs_run = 0

        def evaluate() -> tuple[float, Dict[int, float]]:
            split_tokens = self.val_tokens if use_val else self.train_tokens
            split_targets = silver_val if use_val else silver_train
            predictions = self._predict_matrix(split_tokens, rules)
            per_rule = {}
            values = []
            for idx, rule in enumerate(rules):
                value = _metric_value(predictions[:, idx], split_targets[idx], metric_name)
                per_rule[id(rule)] = value
                values.append(value)
            return float(np.nanmean(values)), per_rule

        LOGGER.info(
            "Joint BERT fine-tune on %d rules (%s, bert_lr=%g, mlp_lr=%g)",
            len(rules),
            settings["strategy"],
            settings["bert_learning_rate"],
            settings["mlp_learning_rate"],
        )
        epoch_inds = list(range(settings["max_epochs"]))
        for epoch in tqdm(epoch_inds, desc="BERT training epochs"):
            self.encoder.train()
            for head in heads:
                head.train()
            order = torch.randperm(silver_train.shape[1], generator=generator)
            batch_starts = list(range(0, silver_train.shape[1], settings["batch_size"]))
            for start in tqdm(batch_starts, desc='BERT training batches'):
                chosen = order[start : start + settings["batch_size"]]
                batch_tokens = _slice_tokens(self.train_tokens, chosen, self.device)
                batch_targets = train_targets[:, chosen].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                cls = _forward_cls(self.encoder, batch_tokens)
                losses = [
                    loss_fn(heads[idx](cls), batch_targets[idx])
                    for idx in range(len(rules))
                ]
                if not all(torch.isfinite(loss) for loss in losses):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                weighter.backward(losses)
                nn.utils.clip_grad_norm_(
                    [param for param in self.encoder.parameters() if param.requires_grad]
                    + [param for head in heads for param in head.parameters()],
                    1.0,
                )
                optimizer.step()
            epochs_run = epoch + 1
            current, per_rule = evaluate()
            LOGGER.info(
                "Dynamic epoch %d %s mean %s=%.4f weights=%s",
                epochs_run,
                "val" if use_val else "train",
                metric_name,
                current,
                [round(value, 4) for value in weighter.weights()],
            )
            if _improved(current, best_metric, direction, settings["min_delta"]):
                best_metric = current
                best_epoch = epoch
                best_encoder = _copy_state(self.encoder)
                best_heads = {id(rule): _copy_state(self._models[id(rule)]) for rule in rules}
                best_rule_metrics = per_rule
                wait = 0
            else:
                wait += 1
                if wait >= settings["patience"]:
                    break

        if best_encoder is None:
            best_encoder = _copy_state(self.encoder)
            best_heads = {id(rule): _copy_state(self._models[id(rule)]) for rule in rules}
            best_metric = float("nan")
            best_epoch = max(epochs_run - 1, 0)
            _, best_rule_metrics = evaluate()
        self.encoder.load_state_dict(best_encoder)
        for rule_id, state in best_heads.items():
            self._models[rule_id].load_state_dict(state)
        self._best_metrics = best_rule_metrics
        self.encoder.eval()
        for head in self._models.values():
            head.eval()
        LOGGER.info(
            "Dynamic fine-tune stopped at epoch %d (best epoch %d, %s mean %s=%.4f)",
            epochs_run,
            best_epoch + 1,
            "val" if use_val else "train",
            metric_name,
            float(best_metric) if best_metric is not None and np.isfinite(best_metric) else float("nan"),
        )

    def _predict_matrix(self, tokens: dict, rules: Sequence) -> np.ndarray:
        import torch

        sample_count = int(tokens["input_ids"].shape[0])
        if not rules:
            return np.zeros((sample_count, 0), dtype=np.float32)
        if sample_count == 0:
            return np.zeros((0, len(rules)), dtype=np.float32)
        self.encoder.eval()
        heads = [self._models[id(rule)] for rule in rules]
        for head in heads:
            head.eval()
        columns: List[list] = [[] for _ in rules]
        batch_size = self.settings["batch_size"]
        with torch.no_grad():
            for start in range(0, sample_count, batch_size):
                batch_tokens = _slice_tokens(tokens, slice(start, start + batch_size), self.device)
                cls = _forward_cls(self.encoder, batch_tokens)
                for idx, head in enumerate(heads):
                    columns[idx].append(head(cls).detach().cpu().numpy())
        stacked = [np.concatenate(parts) for parts in columns]
        predictions = np.stack(stacked, axis=1).astype(np.float32, copy=False)
        return np.clip(predictions, -1.0, 1.0)

    def save_round(self, round_idx: int, output_dir: str, rules: Sequence) -> str:
        import torch

        directory = os.path.join(output_dir, "rule_mlps", f"round_{int(round_idx):02d}")
        os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "state_dict": _copy_state(self.encoder),
                "model_name": self.settings["model_name"],
                "max_length": self.settings["max_length"],
            },
            os.path.join(directory, "bert.pt"),
        )
        manifest_rules = []
        for idx, rule in enumerate(rules):
            module = self._models.get(id(rule))
            if module is None:
                raise RuntimeError(f"No trained MLP for rule {idx} when saving round {round_idx}")
            metric_value = self._best_metrics.get(id(rule))
            metadata = {
                "rule": rule.text,
                "rule_index": idx,
                "source_round": int(getattr(rule, "source_round", -1)),
                "input_dim": self.hidden_size,
                "hidden_dims": list(self.settings["hidden_dims"]),
                "dropout": float(self.settings["dropout"]),
                "metric_name": self.settings["metric"],
                "best_metric": None
                if metric_value is None or not np.isfinite(metric_value)
                else float(metric_value),
            }
            torch.save(
                {"state_dict": _copy_state(module), "metadata": metadata},
                os.path.join(directory, f"rule_{idx:03d}.pt"),
            )
            manifest_rules.append(metadata)
        manifest = {
            "round": int(round_idx),
            "feature_mode": "bert_dynamic",
            "strategy": self.settings["strategy"],
            "model_name": self.settings["model_name"],
            "max_length": self.settings["max_length"],
            "rules": manifest_rules,
        }
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return directory


def predict_saved_dynamic_features(
    *,
    config: dict,
    task,
    frame,
    mlp_dir: str,
    rule_texts: Sequence[str],
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    manifest_path = os.path.join(mlp_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    settings = dynamic_settings(config)
    model_name = str(manifest.get("model_name") or settings["model_name"])
    max_length = int(manifest.get("max_length") or settings["max_length"])
    device = resolve_device(settings["device"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    payload = _torch_load(os.path.join(mlp_dir, "bert.pt"), device)
    encoder.load_state_dict(payload["state_dict"])
    encoder.to(device)
    encoder.eval()
    tokens = tokenize_segments(tokenizer, segments_for_frame(task, frame), max_length)
    sample_count = int(tokens["input_ids"].shape[0])
    with _TRAIN_LOCK:
        heads = load_rule_mlps(mlp_dir, device, expected_rules=rule_texts)
        if not heads:
            return np.zeros((sample_count, 0), dtype=np.float32)
        modules = [head.model for head in heads]
        for module in modules:
            module.eval()
        pieces: List[list] = [[] for _ in modules]
        with torch.no_grad():
            for start in range(0, sample_count, settings["batch_size"]):
                batch_tokens = _slice_tokens(tokens, slice(start, start + settings["batch_size"]), device)
                cls = _forward_cls(encoder, batch_tokens)
                for idx, module in enumerate(modules):
                    pieces[idx].append(module(cls).detach().cpu().numpy())
        columns = [np.concatenate(part) if part else np.zeros((0,), dtype=np.float32) for part in pieces]
    predictions = np.stack(columns, axis=1).astype(np.float32, copy=False)
    return np.clip(predictions, -1.0, 1.0)
