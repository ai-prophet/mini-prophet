"""Evaluation metrics for mini-prophet forecasts."""

from __future__ import annotations

from typing import Protocol


class Metric(Protocol):
    """Protocol for forecast evaluation metrics."""

    name: str

    def compute(self, probabilities: dict[str, float], ground_truth: dict[str, int]) -> float: ...


class BrierScore:
    """Mean squared error between predicted probabilities and binary ground truth."""

    name = "brier_score"

    def compute(self, probabilities: dict[str, float], ground_truth: dict[str, int]) -> float:
        total = 0.0
        n = 0
        for outcome, truth in ground_truth.items():
            pred = probabilities.get(outcome, 0.0)
            total += (pred - truth) ** 2
            n += 1
        return total / n if n else 0.0


_METRIC_REGISTRY: dict[str, Metric] = {}


def register_metric(metric: Metric) -> None:
    _METRIC_REGISTRY[metric.name] = metric


def get_metrics() -> dict[str, Metric]:
    return dict(_METRIC_REGISTRY)


register_metric(BrierScore())


def normalize_ground_truth(raw: dict[str, int] | int | float) -> dict[str, int]:
    """Normalize various ground truth formats to {"Yes": 0|1, "No": 0|1}.

    Accepts:
    - int or float (0 or 1)
    - dict with case-insensitive "Yes" key (e.g. {"yes": 1}, {"YES": 0})
    - dict with both "Yes" and "No" keys (validated as complementary)
    """
    if isinstance(raw, int | float):
        if raw not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Ground truth must be 0 or 1, got {raw}")
        v = int(raw)
        return {"Yes": v, "No": 1 - v}

    if not isinstance(raw, dict):
        raise ValueError(f"Ground truth must be int, float, or dict, got {type(raw).__name__}")

    normalized: dict[str, int] = {}
    for key, val in raw.items():
        k = key.strip().capitalize()
        if k not in ("Yes", "No"):
            raise ValueError(f"Ground truth key must be 'Yes' or 'No', got '{key}'")
        if val not in (0, 1):
            raise ValueError(f"Ground truth value for '{key}' must be 0 or 1, got {val}")
        normalized[k] = val

    if "Yes" not in normalized:
        raise ValueError("Ground truth must include 'Yes' key")

    if "No" not in normalized:
        normalized["No"] = 1 - normalized["Yes"]
    elif normalized["Yes"] + normalized["No"] != 1:
        raise ValueError(
            f"Ground truth values must be complementary (sum to 1), "
            f"got Yes={normalized['Yes']}, No={normalized['No']}"
        )

    return normalized


def validate_ground_truth(ground_truth: dict[str, int]) -> None:
    """Validate that ground_truth is a valid binary ground truth dict (already normalized)."""
    if not isinstance(ground_truth, dict):
        raise ValueError("ground_truth must be a dict")
    if set(ground_truth.keys()) != {"Yes", "No"}:
        raise ValueError(
            f"ground_truth keys must be {{'Yes', 'No'}}, got {set(ground_truth.keys())}"
        )
    for key, val in ground_truth.items():
        if val not in (0, 1):
            raise ValueError(f"ground_truth['{key}'] must be 0 or 1, got {val}")
    if ground_truth["Yes"] + ground_truth["No"] != 1:
        raise ValueError("ground_truth values must be complementary")


def evaluate_submission(
    probabilities: dict[str, float], ground_truth: dict[str, int]
) -> dict[str, float]:
    """Run all registered metrics and return {metric_name: score}."""
    return {
        name: metric.compute(probabilities, ground_truth)
        for name, metric in _METRIC_REGISTRY.items()
    }
