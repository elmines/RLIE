"""Per-rule MLPs on a BERT [CLS] vector.

``regression`` predicts a score in [-1, 1]. ``classification`` predicts three
logits for the labels -1, 0, and 1; the logistic-regression feature is the
argmax, written back as one of those labels.
"""

from __future__ import annotations

import json
from rlie.utils.logger_config import LoggerConfig
import os
import threading
from copy import deepcopy
from typing import Dict, List, Optional, Sequence

import numpy as np

from rlie.bert.store import embedding_dir_for_task, load_embedding_index


LOGGER = LoggerConfig.get_logger("rlie.bert.mlp")
_TRAIN_LOCK = threading.Lock()

_METRIC_DIRECTION = {
    "mse": "min",
    "mae": "min",
    "accuracy": "max",
    "cross_entropy": "min",
}
_HEAD_METRICS = {
    "regression": {"mse", "mae", "accuracy"},
    "classification": {"accuracy", "cross_entropy"},
}


def resolve_head(value) -> str:
    raw = str(value or "regression").strip().lower().replace("-", "_")
    aliases = {
        "regression": "regression",
        "reg": "regression",
        "classification": "classification",
        "classify": "classification",
    }
    if raw not in aliases:
        raise ValueError("bert.mlp.head must be 'regression' or 'classification'")
    return aliases[raw]


def resolve_device(spec: str):
    import torch

    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{spec}' was requested but CUDA is not available")
        index = 0 if device.index is None else device.index
        count = torch.cuda.device_count()
        if index >= count:
            raise RuntimeError(f"CUDA device '{spec}' is out of range; {count} device(s) are visible")
    return device


def mlp_settings(config: Dict) -> Dict:
    bert_cfg = config.get("bert") or {}
    mlp_cfg = dict(bert_cfg.get("mlp") or {})
    hidden = mlp_cfg.get("hidden_dims", [128])
    if isinstance(hidden, int):
        hidden = [hidden]
    mlp_cfg["hidden_dims"] = [int(dim) for dim in hidden]
    mlp_cfg["dropout"] = float(mlp_cfg.get("dropout", 0.1))
    mlp_cfg["learning_rate"] = float(mlp_cfg.get("learning_rate", 1e-3))
    mlp_cfg["weight_decay"] = float(mlp_cfg.get("weight_decay", 1e-4))
    mlp_cfg["batch_size"] = max(1, int(mlp_cfg.get("batch_size", 64)))
    mlp_cfg["max_epochs"] = max(1, int(mlp_cfg.get("max_epochs", 50)))
    mlp_cfg["patience"] = max(1, int(mlp_cfg.get("patience", 5)))
    mlp_cfg["min_delta"] = float(mlp_cfg.get("min_delta", 1e-4))
    head = resolve_head(mlp_cfg.get("head", "regression"))
    default_metric = "cross_entropy" if head == "classification" else "mse"
    metric = str(mlp_cfg.get("metric", default_metric)).strip().lower()
    allowed = _HEAD_METRICS[head]
    if metric not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"bert.mlp.metric for head '{head}' must be one of: {choices}")
    mlp_cfg["head"] = head
    mlp_cfg["metric"] = metric
    return mlp_cfg


def classification_index_numpy(targets: np.ndarray) -> np.ndarray:
    """Map silver labels {-1, 0, 1} to class indices {0, 1, 2}."""
    values = np.rint(np.asarray(targets, dtype=np.float64)).astype(np.int64).reshape(-1)
    if values.size and not np.isin(values, (-1, 0, 1)).all():
        raise ValueError("MLP classification targets must be -1, 0, or 1")
    return values + 1


def classification_index_torch(targets):
    """Map a tensor of silver labels {-1, 0, 1} to class indices {0, 1, 2}."""
    import torch

    values = targets.round().to(dtype=torch.long)
    if values.numel() and bool(((values < -1) | (values > 1)).any()):
        raise ValueError("MLP classification targets must be -1, 0, or 1")
    return values + 1


def features_from_raw(raw: np.ndarray, head: str) -> np.ndarray:
    """Turn a rule MLP output into the logistic-regression feature column."""
    resolved = resolve_head(head)
    array = np.asarray(raw)
    if resolved == "regression":
        return np.clip(array.reshape(-1), -1.0, 1.0).astype(np.float32, copy=False)
    if array.ndim == 1 or array.shape[-1] != 3:
        width = 1 if array.ndim == 1 else int(array.shape[-1])
        raise ValueError(f"classification MLP returned {width} outputs; expected 3 logits")
    return (np.argmax(array, axis=-1).astype(np.float32) - 1.0).reshape(-1)


def rule_criterion(output, targets, head: str, mse_loss, ce_loss):
    """MSE against the silver score, or cross-entropy over the three labels."""
    if resolve_head(head) == "regression":
        return mse_loss(output, targets)
    return ce_loss(output, classification_index_torch(targets))


def build_rule_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float, head: str = "regression"):
    """Feed-forward rule head.

    Regression ends in tanh so every score lies in [-1, 1]. Classification
    returns three logits, ordered as the labels -1, 0, 1.
    """
    from torch import nn

    resolved = resolve_head(head)

    class _RuleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = resolved
            layers: List[nn.Module] = [nn.LayerNorm(input_dim)]
            previous = input_dim
            for hidden in hidden_dims:
                layers.append(nn.Linear(previous, int(hidden)))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                previous = int(hidden)
            if resolved == "classification":
                layers.append(nn.Linear(previous, 3))
            else:
                layers.append(nn.Linear(previous, 1))
                layers.append(nn.Tanh())
            self.backbone = nn.Sequential(*layers)

        def forward(self, inputs):
            output = self.backbone(inputs)
            if self.head == "classification":
                return output
            return output.squeeze(-1)

    return _RuleMLP()


def _improved(current: float, best: Optional[float], direction: str, min_delta: float) -> bool:
    if best is None or not np.isfinite(best):
        return np.isfinite(current)
    if not np.isfinite(current):
        return False
    if direction == "min":
        return best - current > min_delta
    return current - best > min_delta


def _ternary_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    rounded = np.clip(np.rint(predictions), -1, 1)
    if len(targets) == 0:
        return float("nan")
    return float(np.mean(rounded == targets))


def _metric_value(predictions: np.ndarray, targets: np.ndarray, metric: str) -> float:
    if metric == "mse":
        return float(np.mean((predictions - targets) ** 2))
    if metric == "mae":
        return float(np.mean(np.abs(predictions - targets)))
    if metric == "accuracy":
        return _ternary_accuracy(predictions, targets)
    if metric == "cross_entropy":
        raise ValueError("cross_entropy is computed from logits, not from class features")
    raise ValueError(f"Unsupported MLP metric '{metric}'")


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    import torch
    import torch.nn.functional as F

    if len(np.asarray(targets)) == 0:
        return float("nan")
    scores = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    index = torch.from_numpy(classification_index_numpy(targets))
    return float(F.cross_entropy(scores, index).item())


class TrainedRuleMLP:
    def __init__(self, model, device, metadata: Dict):
        self.model = model
        self.device = device
        self.metadata = metadata

    def head_name(self) -> str:
        recorded = (self.metadata or {}).get("head") or getattr(self.model, "head", "regression")
        return resolve_head(recorded)

    def forward_numpy(self, embeddings: np.ndarray, batch_size: int = 256) -> np.ndarray:
        import torch

        if len(embeddings) == 0:
            if self.head_name() == "classification":
                return np.zeros((0, 3), dtype=np.float32)
            return np.zeros((0,), dtype=np.float32)
        self.model.eval()
        chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(embeddings), max(1, batch_size)):
                batch = torch.from_numpy(np.asarray(embeddings[start : start + batch_size], dtype=np.float32))
                batch = batch.to(self.device)
                chunks.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def predict(self, embeddings: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if len(embeddings) == 0:
            return np.zeros((0,), dtype=np.float32)
        return features_from_raw(self.forward_numpy(embeddings, batch_size=batch_size), self.head_name())


def _copy_state(model) -> Dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_rule_mlp(
    embeddings: np.ndarray,
    targets: np.ndarray,
    val_embeddings: Optional[np.ndarray],
    val_targets: Optional[np.ndarray],
    *,
    device,
    mlp_cfg: Dict,
    seed: int,
    rule_text: str,
) -> TrainedRuleMLP:
    import torch
    from torch import nn

    input_dim = int(embeddings.shape[1])
    head = resolve_head(mlp_cfg.get("head", "regression"))
    torch.manual_seed(int(seed))
    model = build_rule_mlp(input_dim, mlp_cfg["hidden_dims"], mlp_cfg["dropout"], head)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mlp_cfg["learning_rate"],
        weight_decay=mlp_cfg["weight_decay"],
    )
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    train_x = np.asarray(embeddings, dtype=np.float32)
    train_y = np.asarray(targets, dtype=np.float32).reshape(-1)
    use_val = val_embeddings is not None and val_targets is not None and len(val_embeddings) > 0
    metric_name = mlp_cfg["metric"]
    direction = _METRIC_DIRECTION[metric_name]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    best_state = None
    best_metric = None
    best_epoch = -1
    wait = 0
    epochs_run = 0

    def evaluate(split_x: np.ndarray, split_y: np.ndarray) -> float:
        holder = TrainedRuleMLP(model, device, {"head": head})
        if metric_name == "cross_entropy":
            return _cross_entropy(holder.forward_numpy(split_x, batch_size=mlp_cfg["batch_size"]), split_y)
        predictions = holder.predict(split_x, batch_size=mlp_cfg["batch_size"])
        return _metric_value(predictions, split_y, metric_name)

    for epoch in range(mlp_cfg["max_epochs"]):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(train_x), mlp_cfg["batch_size"]):
            chosen = order[start : start + mlp_cfg["batch_size"]].numpy()
            batch_x = torch.from_numpy(train_x[chosen]).to(device)
            batch_y = torch.from_numpy(train_y[chosen]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = rule_criterion(model(batch_x), batch_y, head, mse_loss, ce_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        epochs_run = epoch + 1
        if use_val:
            current = evaluate(np.asarray(val_embeddings, dtype=np.float32), np.asarray(val_targets, dtype=np.float32))
        else:
            current = evaluate(train_x, train_y)
        if _improved(current, best_metric, direction, mlp_cfg["min_delta"]):
            best_metric = current
            best_epoch = epoch
            best_state = _copy_state(model)
            wait = 0
        else:
            wait += 1
            if wait >= mlp_cfg["patience"]:
                break

    if best_state is None:
        best_state = _copy_state(model)
        best_metric = float("nan")
        best_epoch = max(epochs_run - 1, 0)
    model.load_state_dict(best_state)
    model.eval()
    metadata = {
        "rule": rule_text,
        "metric_name": metric_name,
        "metric_direction": direction,
        "best_metric": None if best_metric is None or not np.isfinite(best_metric) else float(best_metric),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_run),
        "input_dim": input_dim,
        "hidden_dims": list(mlp_cfg["hidden_dims"]),
        "dropout": float(mlp_cfg["dropout"]),
        "head": head,
        "monitored_split": "val" if use_val else "train",
    }
    LOGGER.info(
        "Rule MLP '%s' stopped at epoch %d (best epoch %d, %s %s=%.4f)",
        rule_text[:80],
        epochs_run,
        best_epoch + 1,
        metadata["monitored_split"],
        metric_name,
        float(best_metric) if best_metric is not None and np.isfinite(best_metric) else float("nan"),
    )
    return TrainedRuleMLP(model, device, metadata)


class RuleMLPBank:
    """One MLP per retained rule, trained on that rule's LLM ternary labels."""

    def __init__(
        self,
        train_embeddings: np.ndarray,
        val_embeddings: Optional[np.ndarray],
        device,
        mlp_cfg: Dict,
        *,
        seed: int,
    ):
        self.train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
        self.val_embeddings = None if val_embeddings is None else np.asarray(val_embeddings, dtype=np.float32)
        self.device = device
        self.mlp_cfg = mlp_cfg
        self.seed = int(seed)
        self._models: Dict[int, TrainedRuleMLP] = {}

    @classmethod
    def from_config(cls, config: Dict, task, task_root: str, train_df, val_df) -> "RuleMLPBank":
        bert_cfg = config.get("bert") or {}
        if not task_root:
            raise ValueError("feature_mode bert_static requires a task id such as real/headline_binary")
        directory = embedding_dir_for_task(bert_cfg.get("embeddings_dir", "embeddings"), task_root)
        index = load_embedding_index(directory)
        expected_model = bert_cfg.get("model_name")
        if expected_model and index.model_name and index.model_name != expected_model:
            raise ValueError(
                f"Embeddings in {directory} were produced by '{index.model_name}', "
                f"but bert.model_name is '{expected_model}'."
            )
        train_embeddings = index.align(task, train_df)
        val_embeddings = None
        if val_df is not None and len(val_df) > 0:
            val_embeddings = index.align(task, val_df)
        device_spec = str(bert_cfg.get("mlp_device") or bert_cfg.get("device") or "cpu")
        device = resolve_device(device_spec)
        LOGGER.info(
            "Loaded BERT embeddings from %s (%d train, %s val, dim=%d) for MLPs on %s",
            directory,
            train_embeddings.shape[0],
            0 if val_embeddings is None else val_embeddings.shape[0],
            train_embeddings.shape[1],
            device,
        )
        return cls(
            train_embeddings,
            val_embeddings,
            device,
            mlp_settings(config),
            seed=int(config.get("data_seed", 0)),
        )

    def sync_and_predict(self, feature_store) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Train any new rule MLPs and return one logistic-regression feature per rule."""
        with _TRAIN_LOCK:
            rules = list(feature_store.rules)
            silver_train = np.asarray(feature_store.features, dtype=np.float32)
            if silver_train.ndim != 2 or silver_train.shape[0] != len(rules):
                raise ValueError("Silver-label matrix does not align with the current rules")
            if silver_train.shape[1] != self.train_embeddings.shape[0]:
                raise ValueError(
                    f"Train embeddings have {self.train_embeddings.shape[0]} rows but "
                    f"silver labels have {silver_train.shape[1]}"
                )
            silver_val = None
            if self.val_embeddings is not None:
                silver_val = np.asarray(feature_store.val_features, dtype=np.float32)
                if silver_val.ndim != 2 or silver_val.shape != (len(rules), self.val_embeddings.shape[0]):
                    raise ValueError("Validation silver labels do not align with validation embeddings")

            alive = {id(rule) for rule in rules}
            for stale in list(self._models):
                if stale not in alive:
                    del self._models[stale]

            for idx, rule in enumerate(rules):
                if id(rule) in self._models:
                    continue
                val_x = self.val_embeddings
                val_y = None if silver_val is None else silver_val[idx]
                self._models[id(rule)] = train_rule_mlp(
                    self.train_embeddings,
                    silver_train[idx],
                    val_x,
                    val_y,
                    device=self.device,
                    mlp_cfg=self.mlp_cfg,
                    seed=self.seed + idx,
                    rule_text=rule.text,
                )

            train_columns = [
                self._models[id(rule)].predict(self.train_embeddings, batch_size=self.mlp_cfg["batch_size"])
                for rule in rules
            ]
            train_matrix = np.stack(train_columns, axis=1) if train_columns else np.zeros((len(self.train_embeddings), 0))
            val_matrix = None
            if self.val_embeddings is not None:
                val_columns = [
                    self._models[id(rule)].predict(self.val_embeddings, batch_size=self.mlp_cfg["batch_size"])
                    for rule in rules
                ]
                val_matrix = np.stack(val_columns, axis=1) if val_columns else np.zeros((len(self.val_embeddings), 0))
            return train_matrix.astype(np.float32, copy=False), None if val_matrix is None else val_matrix.astype(np.float32, copy=False)

    def save_round(self, round_idx: int, output_dir: str, rules: Sequence) -> str:
        import torch

        directory = os.path.join(output_dir, "rule_mlps", f"round_{int(round_idx):02d}")
        os.makedirs(directory, exist_ok=True)
        manifest_rules = []
        for idx, rule in enumerate(rules):
            trained = self._models.get(id(rule))
            if trained is None:
                raise RuntimeError(f"No trained MLP for rule {idx} when saving round {round_idx}")
            payload = {
                "state_dict": _copy_state(trained.model),
                "metadata": deepcopy(trained.metadata),
            }
            payload["metadata"]["rule_index"] = idx
            payload["metadata"]["source_round"] = int(getattr(rule, "source_round", -1))
            torch.save(payload, os.path.join(directory, f"rule_{idx:03d}.pt"))
            manifest_rules.append(payload["metadata"])
        manifest = {
            "round": int(round_idx),
            "feature_mode": "bert_static",
            "head": resolve_head(self.mlp_cfg.get("head", "regression")),
            "rules": manifest_rules,
        }
        with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return directory


def _torch_load(path: str, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_rule_mlps(directory: str, device, expected_rules: Optional[Sequence[str]] = None) -> List[TrainedRuleMLP]:
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Rule MLP manifest not found: {manifest_path}. "
            "This run directory does not contain BERT-MLP checkpoints for the selected round."
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("rules") or []
    loaded_rules = [str(entry.get("rule", "")) for entry in entries]
    if expected_rules is not None and loaded_rules != list(expected_rules):
        raise ValueError(
            f"Saved MLPs in {directory} do not match the rule snapshot "
            f"({len(loaded_rules)} MLP rules vs {len(list(expected_rules))} snapshot rules)."
        )
    models: List[TrainedRuleMLP] = []
    for idx, entry in enumerate(entries):
        path = os.path.join(directory, f"rule_{idx:03d}.pt")
        payload = _torch_load(path, device)
        metadata = dict(payload.get("metadata") or entry)
        model = build_rule_mlp(
            int(metadata["input_dim"]),
            metadata.get("hidden_dims") or [128],
            float(metadata.get("dropout", 0.0)),
            resolve_head(metadata.get("head", "regression")),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        models.append(TrainedRuleMLP(model, device, metadata))
    return models


def predict_saved_mlp_features(
    *,
    config: Dict,
    task,
    task_root: str,
    frame,
    mlp_dir: str,
    rule_texts: Sequence[str],
) -> np.ndarray:
    manifest_path = os.path.join(mlp_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("feature_mode") == "bert_dynamic":
            from rlie.bert.dynamic import predict_saved_dynamic_features

            return predict_saved_dynamic_features(
                config=config,
                task=task,
                frame=frame,
                mlp_dir=mlp_dir,
                rule_texts=rule_texts,
            )
    bert_cfg = config.get("bert") or {}
    directory = embedding_dir_for_task(bert_cfg.get("embeddings_dir", "embeddings"), task_root)
    index = load_embedding_index(directory)
    expected_model = bert_cfg.get("model_name")
    if expected_model and index.model_name and index.model_name != expected_model:
        raise ValueError(
            f"Embeddings in {directory} were produced by '{index.model_name}', "
            f"but bert.model_name is '{expected_model}'."
        )
    embeddings = index.align(task, frame)
    device = resolve_device(str(bert_cfg.get("mlp_device") or bert_cfg.get("device") or "cpu"))
    batch_size = max(1, int((bert_cfg.get("mlp") or {}).get("batch_size", 64)))
    with _TRAIN_LOCK:
        models = load_rule_mlps(mlp_dir, device, expected_rules=rule_texts)
        if models and embeddings.shape[1] != int(models[0].metadata["input_dim"]):
            raise ValueError(
                f"Embedding size {embeddings.shape[1]} does not match the saved MLP input size "
                f"{models[0].metadata['input_dim']}"
            )
        columns = [model.predict(embeddings, batch_size=batch_size) for model in models]
    if not columns:
        return np.zeros((len(frame), 0), dtype=np.float32)
    return np.stack(columns, axis=1).astype(np.float32, copy=False)
