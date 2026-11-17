"""Softmax-regression classifier trained with mini-batch SGD. Mirrors core/SoftmaxModel.kt.

Parameters: W (n_classes x n_features, row-major) and b (n_classes).
With the 12-feature baseline that is 6*12 + 6 = 78 parameters.
"""
import numpy as np


class SoftmaxModel:
    def __init__(self, n_features, n_classes=6):
        self.n_features = n_features
        self.n_classes = n_classes
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

    # ---- inference ----
    def logits(self, X):
        return np.asarray(X) @ self.W.T + self.b

    def predict_proba(self, X):
        z = self.logits(X)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        p = self.predict_proba(X)
        loss = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))
        acc = float(np.mean(p.argmax(axis=1) == y))
        return loss, acc

    # ---- training ----
    def train(self, X, y, epochs=1, lr=0.1, batch_size=16, l2=1e-4, rng=None, shuffle=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = len(y)
        if n == 0:
            return
        for _ in range(epochs):
            order = np.arange(n)
            if shuffle:
                (rng or np.random.default_rng()).shuffle(order)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                xb, yb = X[idx], y[idx]
                p = self.predict_proba(xb)
                p[np.arange(len(yb)), yb] -= 1.0
                gW = p.T @ xb / len(yb)
                gb = p.mean(axis=0)
                self.W -= lr * (gW + l2 * self.W)
                self.b -= lr * gb

    # ---- (de)serialisation in the wire format used by the server and phones ----
    def to_dict(self, feature_set=None):
        d = {"n_features": self.n_features, "n_classes": self.n_classes,
             "W": self.W.reshape(-1).tolist(), "b": self.b.tolist()}
        if feature_set:
            d["feature_set"] = feature_set
        return d

    @classmethod
    def from_dict(cls, d):
        m = cls(d["n_features"], d["n_classes"])
        m.W = np.array(d["W"], dtype=float).reshape(m.n_classes, m.n_features)
        m.b = np.array(d["b"], dtype=float)
        return m

    def copy(self):
        m = SoftmaxModel(self.n_features, self.n_classes)
        m.W = self.W.copy()
        m.b = self.b.copy()
        return m
