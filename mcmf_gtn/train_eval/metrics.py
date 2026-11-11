import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix,
)


def compute_all_metrics(y_true, y_pred, y_prob):
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        y_prob = y_prob.cpu().numpy()
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    n_classes = len(np.unique(y_true))
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}

    if n_classes > 1:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["g_mean"] = float(np.sqrt(metrics["recall"] * specificity))
        metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        if len(np.unique(y_prob[y_true == 1])) > 0 and len(np.unique(y_prob[y_true == 0])) > 0:
            try:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                metrics["auc"] = float("nan")
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
    else:
        for k in ["precision", "recall", "f1", "mcc", "balanced_acc",
                  "g_mean", "fpr", "auc", "pr_auc"]:
            metrics[k] = float("nan")

    return metrics


def paired_ttest(results_a, results_b, metric="f1"):
    vals_a = [r[metric] for r in results_a.values()]
    vals_b = [r[metric] for r in results_b.values()]
    t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
    return {"t_statistic": float(t_stat), "p_value": float(p_val),
            "significant": p_val < 0.05}


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
