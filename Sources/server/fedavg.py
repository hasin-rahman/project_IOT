"""Federated Averaging (McMahan et al., 2017): sample-weighted mean of client parameters."""
import math


def zero_model(n_features, n_classes, feature_set):
    return {
        "n_features": n_features,
        "n_classes": n_classes,
        "feature_set": feature_set,
        "W": [0.0] * (n_features * n_classes),
        "b": [0.0] * n_classes,
    }


def validate_model(model, n_features, n_classes):
    """Raise ValueError if a client model has the wrong shape or non-finite values."""
    W, b = model.get("W"), model.get("b")
    if not isinstance(W, list) or not isinstance(b, list):
        raise ValueError("model must contain lists W and b")
    if len(W) != n_features * n_classes or len(b) != n_classes:
        raise ValueError(f"expected W[{n_features * n_classes}] and b[{n_classes}], "
                         f"got W[{len(W)}] and b[{len(b)}]")
    for v in W + b:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("model contains a non-finite value")


def fedavg(updates):
    """updates: list of (model_dict, num_samples). Returns the averaged W and b."""
    if not updates:
        raise ValueError("no updates to average")
    total = float(sum(n for _, n in updates))
    if total <= 0:
        raise ValueError("total sample count must be positive")
    first = updates[0][0]
    W = [0.0] * len(first["W"])
    b = [0.0] * len(first["b"])
    for model, n in updates:
        w = n / total
        for i, v in enumerate(model["W"]):
            W[i] += w * v
        for i, v in enumerate(model["b"]):
            b[i] += w * v
    return W, b
