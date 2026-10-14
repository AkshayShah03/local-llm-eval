import numpy as np


def cohen_kappa(a, b, labels=None, weights=None) -> float:
    labels = sorted(set(a) | set(b)) if labels is None else list(labels)
    idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    observed = np.zeros((n, n))
    for x, y in zip(a, b, strict=True):
        observed[idx[x], idx[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if weights is None:
        w = 1 - np.eye(n)
    elif weights == "quadratic":
        i, j = np.indices((n, n))
        w = (i - j) ** 2 / (n - 1) ** 2
    else:
        raise ValueError(weights)
    denom = (w * expected).sum()
    return 1.0 if denom == 0 else float(1 - (w * observed).sum() / denom)
