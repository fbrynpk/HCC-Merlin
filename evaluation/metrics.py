import numpy as np
import torch


def compute_f1(preds, labels):
    """Compute binary F1 score from array-like predictions and labels."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    TP = float(((preds == 1) & (labels == 1)).sum())
    FP = float(((preds == 1) & (labels == 0)).sum())
    FN = float(((preds == 0) & (labels == 1)).sum())

    if (TP + FP) == 0 or (TP + FN) == 0:
        return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    denom = precision + recall

    return 2 * (precision * recall) / denom if denom > 0 else 0.0


def bootstrap_f1(preds, labels, n_bootstraps=1000, seed=None):
    """
    Estimate 95% confidence interval for F1 via bootstrap resampling.

    Args:
        preds: Array of predicted binary labels.
        labels: Array of ground-truth binary labels.
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper, bootstrapped_scores): 2.5th/97.5th percentile bounds
        and the full list of per-bootstrap F1 scores.
    """
    rng = np.random.RandomState(seed)
    scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(preds), len(preds))
        scores.append(compute_f1(preds[indices], labels[indices]))

    lower = float(np.percentile(scores, 2.5))
    upper = float(np.percentile(scores, 97.5))
    return lower, upper, scores


def evaluate_predictions(preds, labels):
    """
    Print and return F1 with bootstrap CI.

    Args:
        preds: List or array of integer predictions.
        labels: List or array of integer ground-truth labels.

    Returns:
        (f1, lower, upper, bootstrapped_scores)
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    f1 = compute_f1(preds, labels)
    lower, upper, bootstrapped_scores = bootstrap_f1(preds, labels)

    print(f"F1:       {f1:.4f}")
    print(f"95% CI:   [{lower:.4f}, {upper:.4f}]")

    return f1, lower, upper, bootstrapped_scores


def label_smoothing(targets, smoothing=0.1, mode="binary"):
    """
    Apply label smoothing for binary or multi-class classification.

    Args:
        targets: Ground-truth label tensor.
            - 'binary': shape (N,) or (N, 1) with values in {0, 1}
            - 'multiclass': one-hot tensor of shape (N, C)
        smoothing: Smoothing factor in [0, 1].
        mode: 'binary' or 'multiclass'.

    Returns:
        Smoothed label tensor (same shape as input).
    """
    with torch.no_grad():
        if mode == "binary":
            return targets * (1.0 - smoothing) + (1.0 - targets) * smoothing
        elif mode == "multiclass":
            num_classes = targets.size(-1)
            return targets * (1.0 - smoothing) + smoothing / num_classes
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'binary' or 'multiclass'.")