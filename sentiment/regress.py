"""Day 6 CLI: next-day return regression, split by time, evaluated out of
sample only.

    python -m sentiment.regress
    python -m sentiment.regress --live
    python -m sentiment.regress --train-frac 0.7

Fits a single-predictor OLS regression of **next-day return** (Day 5's
``lagged_return``: the open-to-close return of the trading session after a
headline's leak-free aligned session) on VADER's ``compound`` sentiment
score, using only the headlines whose lagged return could actually be
computed (see Day 5's README Findings on why 8 of 23 could not be).

The train/test split is **chronological, not random**: ``sentiment.
correlate.build_rows`` returns headlines in ``published_at`` order (
``sentiment.headline.write_csv`` always writes oldest-first), so taking the
first ``--train-frac`` fraction as train and the rest as test means train is
genuinely the past relative to test. A random or k-fold split here would let
a later headline's fitted slope leak into an earlier prediction, which is
the same look-ahead Day 4's market-hours alignment exists to prevent, one
level up the pipeline.

The fit itself only ever sees the training split. Every reported predictive
number - out-of-sample R^2 and MAE - comes from the held-out test split only;
in-sample numbers are printed alongside for comparison but are explicitly
not the result, per NEXT_STEPS.md's "evaluated out of sample only".
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows
from sentiment.stats import mean_absolute_error, ols_fit, oos_r_squared, r_squared

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "regression.csv"
DEFAULT_TRAIN_FRAC = 0.7
MIN_TRAIN = 2
MIN_TEST = 1

ROW_FIELDNAMES = ["title", "ticker", "split", "compound", "actual_return", "predicted_return"]


@dataclass(frozen=True)
class TimeSplitResult:
    n_train: int
    n_test: int
    slope: float
    intercept: float
    train_r2: float
    test_r2: float
    train_mae: float
    test_mae: float
    baseline_test_mae: float


def usable_rows(rows: list[dict]) -> list[dict]:
    """Rows with a next-day (lagged) return to regress against, in the
    chronological order ``build_rows`` already produces."""
    return [r for r in rows if r["lagged_return"] is not None]


def time_split(rows: list[dict], train_frac: float) -> tuple[list[dict], list[dict]]:
    """Split ``rows`` (already time-ordered) into a leading train block and a
    trailing test block. Raises ``ValueError`` if either side would end up
    smaller than the minimum needed to fit or evaluate a line."""
    n = len(rows)
    n_train = int(n * train_frac)
    n_train = max(n_train, MIN_TRAIN)
    n_test = n - n_train
    if n_train < MIN_TRAIN or n_test < MIN_TEST:
        raise ValueError(
            f"only {n} usable row(s) - not enough for a {MIN_TRAIN}+/{MIN_TEST}+ time split"
        )
    return rows[:n_train], rows[n_train:]


def fit_and_evaluate(train: list[dict], test: list[dict]) -> TimeSplitResult:
    train_x = [r["compound"] for r in train]
    train_y = [r["lagged_return"] for r in train]
    test_x = [r["compound"] for r in test]
    test_y = [r["lagged_return"] for r in test]

    fit = ols_fit(train_x, train_y)
    train_pred = [fit.predict(x) for x in train_x]
    test_pred = [fit.predict(x) for x in test_x]

    train_mean = sum(train_y) / len(train_y)
    baseline_test_pred = [train_mean] * len(test_y)

    return TimeSplitResult(
        n_train=len(train),
        n_test=len(test),
        slope=fit.slope,
        intercept=fit.intercept,
        train_r2=r_squared(train_y, train_pred),
        test_r2=oos_r_squared(test_y, test_pred, train_mean),
        train_mae=mean_absolute_error(train_y, train_pred),
        test_mae=mean_absolute_error(test_y, test_pred),
        baseline_test_mae=mean_absolute_error(test_y, baseline_test_pred),
    )


def write_rows_csv(train: list[dict], test: list[dict], fit_slope: float, fit_intercept: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES)
        writer.writeheader()
        for split, group in (("train", train), ("test", test)):
            for r in group:
                writer.writerow(
                    {
                        "title": r["title"],
                        "ticker": r["ticker"],
                        "split": split,
                        "compound": r["compound"],
                        "actual_return": r["lagged_return"],
                        "predicted_return": fit_intercept + fit_slope * r["compound"],
                    }
                )


def run(in_path: Path, out_path: Path, live: bool, train_frac: float) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    rows, _ = build_rows(in_path, live=live)
    rows = usable_rows(rows)
    if not rows:
        print("no headlines have a next-day return yet; nothing to regress", file=sys.stderr)
        return 1

    try:
        train, test = time_split(rows, train_frac)
    except ValueError as exc:
        print(f"cannot evaluate out of sample: {exc}", file=sys.stderr)
        return 1

    result = fit_and_evaluate(train, test)
    write_rows_csv(train, test, result.slope, result.intercept, out_path)

    print(f"{len(rows)} headlines have a next-day return; split by time -> {out_path}")
    print(f"train: n={result.n_train}  (earliest published headlines)")
    print(f"test:  n={result.n_test}  (latest published headlines, held out)")
    print(f"fit (train only): next_day_return = {result.intercept:+.4f} + {result.slope:+.4f} * compound")
    print(f"in-sample  (train): R^2={result.train_r2:+.3f}  MAE={result.train_mae:.4f}")
    print(
        f"out-of-sample (test): R^2={result.test_r2:+.3f}  MAE={result.test_mae:.4f}  "
        f"(baseline 'predict train mean' MAE={result.baseline_test_mae:.4f})"
    )
    if result.test_r2 <= 0:
        print(
            "out-of-sample R^2 <= 0: the fitted line does no better than predicting the "
            "training mean for every held-out headline - no out-of-sample predictive power found."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="per-headline CSV to write")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--train-frac",
        type=float,
        default=DEFAULT_TRAIN_FRAC,
        help="fraction of the (time-ordered) usable headlines to train on; the rest is the held-out test split",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.live, args.train_frac))


if __name__ == "__main__":
    main()
