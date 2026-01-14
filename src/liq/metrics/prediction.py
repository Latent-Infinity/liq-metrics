"""Prediction metrics for classification and regression."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def summarize_classification(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    """Summarize accuracy and macro-F1 for discrete labels.

    Args:
        y_true: Iterable of integer true labels.
        y_pred: Iterable of integer predicted labels.

    Returns:
        Dictionary with count, accuracy, and macro_f1.
    """
    true_list = list(y_true)
    pred_list = list(y_pred)

    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_list:
        return {"count": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    for idx, (yt, yp) in enumerate(zip(true_list, pred_list)):
        if not isinstance(yt, int) or isinstance(yt, bool):
            raise TypeError(f"y_true must be int labels, got {type(yt).__name__} at {idx}")
        if not isinstance(yp, int) or isinstance(yp, bool):
            raise TypeError(f"y_pred must be int labels, got {type(yp).__name__} at {idx}")

    correct = sum(1 for yt, yp in zip(true_list, pred_list) if yt == yp)
    accuracy = correct / len(true_list)

    labels = sorted(set(true_list) | set(pred_list))
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for yt, yp in zip(true_list, pred_list):
        if yt == yp:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1

    f1_sum = 0.0
    for label in labels:
        tp_val = tp[label]
        fp_val = fp[label]
        fn_val = fn[label]
        precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        recall = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_sum += f1

    macro_f1 = f1_sum / len(labels) if labels else 0.0

    return {
        "count": float(len(true_list)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def summarize_regression(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    y_log_var: Iterable[float] | None = None,
    coverage_sigmas: Iterable[int] = (1, 2),
) -> dict[str, float]:
    """Summarize regression metrics.

    Args:
        y_true: Iterable of true values.
        y_pred: Iterable of predicted means.
        y_log_var: Optional iterable of log variances for Gaussian NLL/coverage.
        coverage_sigmas: Sigma levels to compute coverage for.

    Returns:
        Dictionary with count, correlation, nll, and coverage metrics.
    """
    true_list = list(y_true)
    pred_list = list(y_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred must have the same length")

    if not true_list:
        return {"count": 0.0, "corr": 0.0, "nll": 0.0}

    for idx, (yt, yp) in enumerate(zip(true_list, pred_list)):
        if isinstance(yt, bool) or isinstance(yp, bool):
            raise TypeError("y_true and y_pred must be numeric")
        if not isinstance(yt, (int, float)):
            raise TypeError(f"y_true must be numeric, got {type(yt).__name__} at {idx}")
        if not isinstance(yp, (int, float)):
            raise TypeError(f"y_pred must be numeric, got {type(yp).__name__} at {idx}")

    corr = _pearson_corr(true_list, pred_list)
    metrics: dict[str, float] = {
        "count": float(len(true_list)),
        "corr": corr,
    }

    if y_log_var is not None:
        log_var_list = list(y_log_var)
        if len(log_var_list) != len(true_list):
            raise ValueError("y_log_var must match y_true length")
        for idx, lv in enumerate(log_var_list):
            if isinstance(lv, bool) or not isinstance(lv, (int, float)):
                raise TypeError(f"y_log_var must be numeric, got {type(lv).__name__} at {idx}")

        metrics["nll"] = _gaussian_nll(true_list, pred_list, log_var_list)

        for sigma in coverage_sigmas:
            if sigma <= 0:
                raise ValueError("coverage_sigmas must be positive")
            coverage = _coverage(true_list, pred_list, log_var_list, sigma)
            metrics[f"coverage_{sigma}sigma"] = coverage
    else:
        metrics["nll"] = 0.0

    return metrics


def _pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    mean_x = math.fsum(x) / len(x)
    mean_y = math.fsum(y) / len(y)
    cov = math.fsum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = math.fsum((xi - mean_x) ** 2 for xi in x)
    var_y = math.fsum((yi - mean_y) ** 2 for yi in y)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def _gaussian_nll(x: list[float], mean: list[float], log_var: list[float]) -> float:
    total = 0.0
    for xt, mu, lv in zip(x, mean, log_var):
        var = math.exp(lv)
        total += 0.5 * (lv + ((xt - mu) ** 2) / var + math.log(2 * math.pi))
    return total / len(x)


def _coverage(
    x: list[float],
    mean: list[float],
    log_var: list[float],
    sigma: int,
) -> float:
    count = 0
    for xt, mu, lv in zip(x, mean, log_var):
        std = math.exp(0.5 * lv)
        if abs(xt - mu) <= sigma * std:
            count += 1
    return count / len(x)
