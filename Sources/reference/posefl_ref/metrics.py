"""Evaluation metrics. Mirrors core/Metrics.kt."""
import numpy as np


def confusion(y_true, y_pred, n_classes=6):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm  # rows = true class, columns = predicted class


def summary(y_true, y_pred, n_classes=6):
    cm = confusion(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(float)
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    present = support > 0
    return {
        "accuracy": float(tp.sum() / max(1, cm.sum())),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion": cm.tolist(),
    }
