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
class OlsFit:
    slope: float
    intercept: float

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * x


def ols_fit(xs: list[float], ys: list[float]) -> OlsFit:
    """Ordinary least squares for y = intercept + slope * x.

    Closed-form single-predictor fit (no numpy/scipy dependency) - the
    normal equations for one predictor reduce to slope = cov(x,y)/var(x).
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length")
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 points to fit a line")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        # every x identical (Day 5 found this happens often here - see
        # tickers.py's docstring): no information to fit a slope from.
        return OlsFit(slope=0.0, intercept=mean_y)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return OlsFit(slope=slope, intercept=intercept)


def r_squared(y_true: list[float], y_pred: list[float]) -> float:
    """In-sample R^2: 1 - SS_res / SS_tot, SS_tot against y_true's own mean."""
    n = len(y_true)
    mean_y = sum(y_true) / n
    ss_tot = sum((y - mean_y) ** 2 for y in y_true)
    if ss_tot == 0:
        return 0.0
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot


def oos_r_squared(y_test: list[float], y_pred: list[float], train_mean: float) -> float:
    """Out-of-sample R^2 (Campbell-Thompson / Goyal-Welch style): 1 - SS_res /
    SS_tot, where SS_tot benchmarks against the *training* mean, not the
    test set's own mean - the fair comparison is against the naive forecast
    a trader could actually have made ahead of time (predict the historical
    average return), not against a mean that peeks at the test data itself.

    Positive means the model beats that naive baseline out of sample;
    negative means it is worse than just guessing the training mean.
    """
    ss_tot = sum((y - train_mean) ** 2 for y in y_test)
    if ss_tot == 0:
        return 0.0
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_test, y_pred))
    return 1 - ss_res / ss_tot


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    n = len(y_true)
    if n == 0:
        raise ValueError("need at least 1 point")
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def percentile_ci(values: list[float], ci: float = 0.95) -> tuple[float, float]:
    """Two-sided percentile interval of ``values`` - e.g. the empirical null
    distribution a permutation/shuffle test produces. Not model-based, so it
    needs no assumption about the shape of ``values``' distribution."""
    if not values:
        raise ValueError("need at least one value")
    tail = (1 - ci) / 2
    ordered = sorted(values)
    n = len(ordered)
    lo = ordered[int(tail * n)]
    hi = ordered[min(int((1 - tail) * n), n - 1)]
    return lo, hi


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
