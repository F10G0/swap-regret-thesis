import math

import numpy as np
from scipy.stats import t as student_t


def mean_confidence_interval_half_width(
    values,
    confidence_level: float = 0.95,
    axis: int = 0,
):
    """Return a two-sided Student-t interval half-width for a sample mean."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim == 0:
        raise ValueError("confidence interval values must have a sample axis")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")

    sample_size = samples.shape[axis]
    if sample_size == 0:
        raise ValueError("at least one value is required")
    if sample_size == 1:
        return np.zeros_like(np.mean(samples, axis=axis), dtype=float)

    critical_value = float(
        student_t.ppf((1.0 + confidence_level) / 2.0, df=sample_size - 1)
    )
    return (
        critical_value
        * np.std(samples, axis=axis, ddof=1)
        / math.sqrt(sample_size)
    )
