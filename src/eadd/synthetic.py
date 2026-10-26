# SPDX-FileCopyrightText: 2026 Nusrat Begum <rupunzelnusrat5@gmail.com>
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Synthetic feature-drift streams for demos and tests.

Each stream has ``n_features`` Gaussian features. Drift moves the mean of the
first ``n_drifting`` features by ``shift`` standard deviations, following one
of four temporal patterns.
"""

from typing import Iterator, List, Tuple

import numpy as np

PATTERNS = ("stable", "abrupt", "gradual", "incremental", "recurring")


def drift_stream(
    pattern: str,
    n_samples: int = 6000,
    drift_at: int = 3000,
    width: int = 1000,
    n_features: int = 5,
    n_drifting: int = 2,
    shift: float = 1.0,
    seed: int = 0,
) -> Tuple[Iterator[dict], List[int]]:
    """Return (stream of {f0..fn: value} dicts, ground-truth drift starts)."""
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}")
    rng = np.random.default_rng(seed)
    names = [f"f{i}" for i in range(n_features)]

    def mean_at(t: int, u: float) -> float:
        if pattern == "stable" or t < drift_at:
            return 0.0
        if pattern == "abrupt":
            return shift
        if pattern == "incremental":
            return shift * min(1.0, (t - drift_at) / width)
        if pattern == "gradual":
            # Samples come from the new concept with growing probability.
            return shift if u < min(1.0, (t - drift_at) / width) else 0.0
        # recurring: new concept for `width` samples, then back, then again
        phase = ((t - drift_at) // width) % 2
        return shift if phase == 0 else 0.0

    if pattern == "stable":
        truth: List[int] = []
    elif pattern == "recurring":
        truth = list(range(drift_at, n_samples, width))
    else:
        truth = [drift_at]

    def gen() -> Iterator[dict]:
        for t in range(n_samples):
            x = rng.normal(size=n_features)
            x[:n_drifting] += mean_at(t, rng.random())
            yield dict(zip(names, x.tolist()))

    return gen(), truth
