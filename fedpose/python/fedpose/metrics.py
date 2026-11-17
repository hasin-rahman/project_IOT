from __future__ import annotations

import numpy as np

from .constants import N_CLASSES


def confusion_matrix(y_true, y_pred, k: int = N_CLASSES) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def summarize(cm: np.ndarray) -> dict:
    """Accuracy, per-class recall, macro-F1 from a confusion matrix.

    Classes with zero support get recall 0 and are still included in macro-F1 (F1 = 0),
    which is the conservative choice when a model never predicts a class.
    """
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "n": int(total),
    }


def evaluate(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    out = summarize(cm)
    out["confusion"] = cm.tolist()
    return out
