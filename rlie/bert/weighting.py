"""Balance one loss term per rule while fine-tuning BERT.

``adaptive_weighting`` starts from uniform weights and, after every batch, sets
each weight to that term's share of the total loss. ``gradnorm`` uses the
``gradnorm-pytorch`` implementation and reads gradient norms from the last
feedforward layer of BERT.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def last_feedforward_parameter(model):
    """Return the weight parameter of the last BERT feedforward projection."""
    from torch import nn

    seen: set[int] = set()
    queue = [model]
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        try:
            return _ffn_output_weight(_last_transformer_layer(current))
        except ValueError:
            pass
        for name in ("base_model", "bert", "modernbert", "model"):
            child = getattr(current, name, None)
            if isinstance(child, nn.Module) and child is not current:
                queue.append(child)
    raise ValueError(
        "Could not find the last feedforward layer of the BERT encoder. "
        "GradNorm needs that layer's weight as grad_norm_parameters."
    )


def _last_transformer_layer(model):
    layers = getattr(model, "layers", None)
    if layers is not None and len(layers) > 0:
        return layers[-1]
    encoder = getattr(model, "encoder", None)
    encoder_layers = getattr(encoder, "layer", None) if encoder is not None else None
    if encoder_layers is not None and len(encoder_layers) > 0:
        return encoder_layers[-1]
    raise ValueError("encoder has no transformer layers")


def _ffn_output_weight(layer):
    from torch import nn

    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        mlp = getattr(layer, "ffn", None)
    if isinstance(mlp, nn.Module):
        for name in ("Wo", "down_proj", "fc2", "dense_2", "w2"):
            linear = getattr(mlp, name, None)
            if isinstance(linear, nn.Linear):
                return linear.weight
        linears = [module for module in mlp.modules() if isinstance(module, nn.Linear)]
        if linears:
            return linears[-1].weight
    output = getattr(layer, "output", None)
    dense = getattr(output, "dense", None) if output is not None else None
    if isinstance(dense, nn.Linear):
        return dense.weight
    raise ValueError("transformer layer has no feedforward projection")


class AdaptiveLossWeighter:
    """Uniform weights at the start of a round, then loss-proportional weights."""

    def __init__(self, num_losses: int):
        if num_losses <= 0:
            raise ValueError("Adaptive weighting needs at least one loss term")
        self._weights = np.full(num_losses, 1.0 / num_losses, dtype=np.float64)

    def weights(self) -> list[float]:
        return [float(value) for value in self._weights]

    def backward(self, losses: Sequence):
        if len(losses) != len(self._weights):
            raise ValueError(
                f"Expected {len(self._weights)} loss terms, received {len(losses)}"
            )
        total = losses[0] * float(self._weights[0])
        for weight, loss in zip(self._weights[1:], losses[1:]):
            total = total + loss * float(weight)
        total.backward()
        self._update(losses)
        return total.detach()

    def _update(self, losses: Sequence) -> None:
        values = []
        for loss in losses:
            value = float(loss.detach())
            if not np.isfinite(value) or value < 0:
                return
            values.append(value)
        denom = float(sum(values))
        if denom <= 0 or not np.isfinite(denom):
            self._weights = np.full(len(values), 1.0 / len(values), dtype=np.float64)
            return
        self._weights = np.asarray([value / denom for value in values], dtype=np.float64)


class GradNormLossWeighter:
    """Library GradNorm balancer over the shared BERT feedforward weight."""

    def __init__(self, num_losses: int, grad_norm_parameter, *, learning_rate: float, alpha: float):
        if num_losses <= 0:
            raise ValueError("GradNorm needs at least one loss term")
        self._single = num_losses == 1
        self._weighter = None
        if self._single:
            return
        try:
            from gradnorm_pytorch import GradNormLossWeighter as LibraryWeighter
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "bert.dynamic.strategy gradnorm requires the gradnorm-pytorch package."
            ) from exc
        self._weighter = LibraryWeighter(
            num_losses=num_losses,
            learning_rate=float(learning_rate),
            restoring_force_alpha=float(alpha),
            grad_norm_parameters=grad_norm_parameter,
        )
        self._weighter.train()

    def weights(self) -> list[float]:
        if self._weighter is None:
            return [1.0]
        return [float(value) for value in self._weighter.loss_weights.detach().cpu()]

    def backward(self, losses: Sequence):
        if self._weighter is None:
            losses[0].backward()
            return
        # The library applies the weighted backward and the GradNorm update
        # itself, and returns nothing once the weights have been stepped.
        self._weighter([loss for loss in losses])


def build_loss_weighter(strategy: str, num_losses: int, grad_norm_parameter, settings: dict):
    if strategy == "adaptive_weighting":
        return AdaptiveLossWeighter(num_losses)
    if strategy == "gradnorm":
        return GradNormLossWeighter(
            num_losses,
            grad_norm_parameter,
            learning_rate=float(settings["gradnorm_learning_rate"]),
            alpha=float(settings["gradnorm_alpha"]),
        )
    raise ValueError("bert.dynamic.strategy must be 'adaptive_weighting' or 'gradnorm'")
