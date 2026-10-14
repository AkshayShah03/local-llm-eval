import numpy as np


def percentile(values, p: float) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.percentile(vals, p, method="linear"))
