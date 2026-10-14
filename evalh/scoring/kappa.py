import numpy as np


def cohen_kappa(a, b, labels=None, weights=None) -> float:
    """Cohen's kappa for two raters. Returns nan when kappa is undefined (expected disagreement is 0)."""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    if len(a) == 0:
        raise ValueError("cannot compute kappa on empty input")
    labels = sorted(set(a) | set(b)) if labels is None else list(labels)
    idx = {label: i for i, label in enumerate(labels)}
    for x in set(a) | set(b):
        if x not in idx:
            raise ValueError(f"label {x!r} is not in labels")
    n = len(labels)
    if weights not in (None, "quadratic"):
        raise ValueError(weights)
    if n == 1:
        return float("nan")
    observed = np.zeros((n, n))
    for x, y in zip(a, b, strict=True):
        observed[idx[x], idx[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if weights is None:
        w = 1 - np.eye(n)
    else:
        i, j = np.indices((n, n))
        w = (i - j) ** 2 / (n - 1) ** 2
    denom = (w * expected).sum()
    if abs(denom) < 1e-12:
        return float("nan")
    return float(1 - (w * observed).sum() / denom)
