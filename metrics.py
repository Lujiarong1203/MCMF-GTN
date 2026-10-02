import math

import numpy as np


def _ranking_metrics(labels, scores):
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    if not positives or not negatives:
        return None, None
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True]) + 1
    start, positive_rank_sum = 0, 0.0
    for end in ends:
        positive_rank_sum += sorted_labels[start:end].sum() * ((start + 1 + end) / 2.0)
        start = int(end)
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    descending_labels, descending_scores = sorted_labels[::-1], sorted_scores[::-1]
    ends = np.flatnonzero(np.r_[descending_scores[1:] != descending_scores[:-1], True]) + 1
    cumulative_positive = np.cumsum(descending_labels)
    previous_positive, ap = 0, 0.0
    for end in ends:
        current_positive = int(cumulative_positive[end - 1])
        ap += (current_positive - previous_positive) * current_positive / end / positives
        previous_positive = current_positive
    return float(auc), float(ap)


def compute_metrics(labels, scores, predictions=None):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("metrics require matching nonempty binary labels and positive scores")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be finite probabilities in [0,1]")
    predictions = (scores >= 0.5).astype(int) if predictions is None else np.asarray(predictions)
    if predictions.shape != labels.shape or not np.isin(predictions, [0, 1]).all():
        raise ValueError("predictions must be matching binary labels")
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (recall + specificity) / 2 if tp + fn and specificity is not None else None
    g_mean = math.sqrt(recall * specificity) if balanced is not None else None
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    auc, ap = _ranking_metrics(labels.astype(int), scores)
    return {
        "accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "roc_auc": auc, "pr_auc": ap,
        "mcc": (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0,
        "balanced_accuracy": balanced, "g_mean": g_mean,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "n": len(labels),
    }


def summarize_metrics(records):
    if not records:
        raise ValueError("no runs to summarize")
    summary = {}
    for name in sorted(set().union(*(record.keys() for record in records))):
        values = [record[name] for record in records if record.get(name) is not None and isinstance(record[name], (int, float))]
        summary[name] = {"mean": float(np.mean(values)) if values else None, "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None, "runs": len(values)}
    return summary
