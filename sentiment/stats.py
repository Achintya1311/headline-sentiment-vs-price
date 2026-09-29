"""Small, dependency-free stats helpers for Day 5's correlation and event study.

No scipy in this project (see requirements.txt) - these are the two things
Day 5 actually needs: a Pearson correlation with a confidence interval, and
a bootstrap confidence interval for a difference in group means. Both are
plain enough to implement directly rather than adding a new dependency for
two functions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# z critical value for a 95% two-sided normal confidence interval.
Z_95 = 1.959963985


@dataclass(frozen=True)
class PearsonResult:
    r: float
    n: int
    ci_low: float
    ci_high: float


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def pearson_with_ci(xs: list[float], ys: list[float]) -> PearsonResult:
    """Pearson r with a 95% CI via the Fisher z-transform. Needs n >= 4 for
    the transform's variance (1/(n-3)) to be defined; below that the CI is
    reported as (-1, 1) rather than a divide-by-zero, because n that small
    tells you almost nothing about the interval anyway."""
    n = len(xs)
    r = pearson_r(xs, ys)
    if n < 4:
        return PearsonResult(r=r, n=n, ci_low=-1.0, ci_high=1.0)
    r_clamped = max(min(r, 1 - 1e-10), -1 + 1e-10)
    z = math.atanh(r_clamped)
    se = 1 / math.sqrt(n - 3)
    lo = math.tanh(z - Z_95 * se)
    hi = math.tanh(z + Z_95 * se)
    return PearsonResult(r=r, n=n, ci_low=lo, ci_high=hi)


@dataclass(frozen=True)
class BootstrapDiffResult:
    diff: float
    ci_low: float
    ci_high: float
    n_a: int
    n_b: int


def bootstrap_mean_diff_ci(
    group_a: list[float],
    group_b: list[float],
    n_boot: int = 5000,
    seed: int = 0,
) -> BootstrapDiffResult:
    """95% percentile bootstrap CI for mean(group_a) - mean(group_b).

    Deterministic (fixed seed) so the CLI and its tests see the same numbers
    on every run. With the small samples this project actually has, this
    interval is expected to be wide - that width is the honest result, not a
    bug to hide.
    """
    if not group_a or not group_b:
        raise ValueError("both groups need at least one observation")
    rng = random.Random(seed)
    diff = sum(group_a) / len(group_a) - sum(group_b) / len(group_b)
    diffs = []
    for _ in range(n_boot):
        sample_a = [group_a[rng.randrange(len(group_a))] for _ in group_a]
        sample_b = [group_b[rng.randrange(len(group_b))] for _ in group_b]
        diffs.append(sum(sample_a) / len(sample_a) - sum(sample_b) / len(sample_b))
    diffs.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = int(0.975 * n_boot) - 1
    return BootstrapDiffResult(
        diff=diff,
        ci_low=diffs[lo_idx],
        ci_high=diffs[hi_idx],
        n_a=len(group_a),
        n_b=len(group_b),
    )
