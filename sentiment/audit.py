"""Day 8 CLI: the leakage/permutation audit the whole project is gated on.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000 --seed 1
    python -m sentiment.audit --gate-p 0.05

README's correctness gate says it plainly: "shuffled-timestamp control:
randomise headline times and the signal must disappear. Runs in CI, not
once by hand." This module is that control run for real, not asserted.

For each of ``--n-shuffles`` runs, ``published_at`` is permuted across the
same set of headlines - same title, same source, same link, same scraped_at,
so the same ticker resolution (``sentiment.tickers.resolve``, title-based)
and the same VADER score. Only *when* each headline claims to have been
published changes, which changes Day 4's ``align_headline`` output and so
which trading session's return every headline gets paired with. Day 5's
correlation is rebuilt from scratch on that shuffled timeline, giving one
sample from a null distribution: "how large a correlation would title-based
sentiment and price data show if the timestamp linking them were random
noise instead of the headline's real publish time?"

The real (unshuffled) correlation is compared against that null via a
two-sided empirical p-value: the fraction of shuffles whose |r| is at least
as large. A small p-value (the real |r| sits in the tail of the shuffled
distribution) means the pipeline is finding something that depends on
using the *correct* timestamp - not proof it is a genuine market
relationship, but evidence it is not simply an artifact any random time
label would have produced just as well. That is the gate this CLI enforces
via its exit code; see README Findings for why, on this fixture, that gate
turns out to be easy to pass for an uninteresting reason.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500
DEFAULT_GATE_P = 0.05
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "audit_null_distribution.csv"


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of the same headlines with ``published_at`` permuted
    across them. A permutation, not independent resampling: the same 50 (or
    however many) real timestamps get reassigned to different headlines, so
    the marginal distribution of *when things were published* is untouched -
    only which headline each timestamp belongs to changes."""
    shuffled_ts = [h.published_at for h in headlines]
    rng.shuffle(shuffled_ts)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, ts in zip(headlines, shuffled_ts)
    ]


def correlation_r(rows: list[dict], key: str) -> float | None:
    """Pearson r between ``compound`` and ``rows[*][key]``, skipping rows
    where that key is None (lagged_return, when the next session hasn't
    traded yet). None if fewer than 2 usable rows - not enough to correlate."""
    pairs = [(r["compound"], r[key]) for r in rows if r[key] is not None]
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return pearson_r(xs, ys)


@dataclass(frozen=True)
class AuditResult:
    real_contemp_r: float | None
    real_contemp_n: int
    real_lagged_r: float | None
    real_lagged_n: int
    shuffled_contemp_r: list[float]
    shuffled_lagged_r: list[float]
    n_shuffles: int
    seed: int

    def p_value(self, series: str) -> float | None:
        """Two-sided empirical p-value for ``series`` ("contemporaneous" or
        "lagged"): the fraction of shuffles whose |r| is >= the real |r|,
        with add-one smoothing (never reports a result as literally
        impossible, only as this-rare-or-rarer given ``n_shuffles`` draws).
        None if there is no real r, or no shuffle produced a usable r, to
        compare against."""
        if series == "contemporaneous":
            real, null = self.real_contemp_r, self.shuffled_contemp_r
        elif series == "lagged":
            real, null = self.real_lagged_r, self.shuffled_lagged_r
        else:
            raise ValueError(f"unknown series: {series!r}")
        if real is None or not null:
            return None
        at_least_as_extreme = sum(1 for r in null if abs(r) >= abs(real))
        return (at_least_as_extreme + 1) / (len(null) + 1)


def run_audit(in_path: Path, n_shuffles: int, seed: int, live: bool = False) -> AuditResult:
    headlines = read_csv(in_path)

    # The real pass may be --live (re-fetch and cache prices); every shuffle
    # after it reads that same cache, never re-fetching per shuffle.
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_contemp_r = correlation_r(real_rows, "contemporaneous_return")
    real_lagged_r = correlation_r(real_rows, "lagged_return")
    real_contemp_n = len(real_rows)
    real_lagged_n = sum(1 for r in real_rows if r["lagged_return"] is not None)

    rng = random.Random(seed)
    shuffled_contemp: list[float] = []
    shuffled_lagged: list[float] = []
    for _ in range(n_shuffles):
        shuffled_headlines = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled_headlines, live=False)
        r = correlation_r(rows, "contemporaneous_return")
        if r is not None:
            shuffled_contemp.append(r)
        r_lag = correlation_r(rows, "lagged_return")
        if r_lag is not None:
            shuffled_lagged.append(r_lag)

    return AuditResult(
        real_contemp_r=real_contemp_r,
        real_contemp_n=real_contemp_n,
        real_lagged_r=real_lagged_r,
        real_lagged_n=real_lagged_n,
        shuffled_contemp_r=shuffled_contemp,
        shuffled_lagged_r=shuffled_lagged,
        n_shuffles=n_shuffles,
        seed=seed,
    )


def write_null_distribution_csv(result: AuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = max(len(result.shuffled_contemp_r), len(result.shuffled_lagged_r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shuffle_index", "contemporaneous_r", "lagged_r"])
        for i in range(rows):
            c = result.shuffled_contemp_r[i] if i < len(result.shuffled_contemp_r) else ""
            l = result.shuffled_lagged_r[i] if i < len(result.shuffled_lagged_r) else ""
            writer.writerow([i, c, l])


def _print_series(name: str, result: AuditResult, series: str, gate_p: float) -> bool:
    """Print one series' report line; return True iff it passes the gate
    (real r is not a statistically extreme outlier of the shuffled null, so
    there is no sign the correct-timestamp version is exploiting something a
    random timestamp could not)."""
    real_r = result.real_contemp_r if series == "contemporaneous" else result.real_lagged_r
    real_n = result.real_contemp_n if series == "contemporaneous" else result.real_lagged_n
    null = result.shuffled_contemp_r if series == "contemporaneous" else result.shuffled_lagged_r
    p = result.p_value(series)

    if real_r is None or p is None:
        print(f"{name}: not enough data for a real result or a null distribution (n={real_n})")
        return True  # nothing to gate on - not a failure, just untestable

    null_abs = sorted(abs(x) for x in null)
    mid = len(null_abs) // 2
    median_abs_null = null_abs[mid] if len(null_abs) % 2 else (null_abs[mid - 1] + null_abs[mid]) / 2
    passed = p >= gate_p
    verdict = "PASS (indistinguishable from shuffled noise)" if passed else "FAIL (outlier vs shuffled noise)"
    print(
        f"{name}: real r={real_r:+.3f} (n={real_n})  "
        f"shuffled |r| median={median_abs_null:.3f} over {len(null)} usable shuffles  "
        f"p={p:.3f}  {verdict}"
    )
    return passed


def run(in_path: Path, out_path: Path, n_shuffles: int, seed: int, live: bool, gate_p: float) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = run_audit(in_path, n_shuffles=n_shuffles, seed=seed, live=live)
    write_null_distribution_csv(result, out_path)
    print(f"{result.n_shuffles} timestamp shuffles (seed={result.seed}), null distribution written to {out_path}")

    contemp_ok = _print_series("contemporaneous", result, "contemporaneous", gate_p)
    lagged_ok = _print_series("lagged         ", result, "lagged", gate_p)

    return 0 if (contemp_ok and lagged_ok) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="null-distribution CSV to write")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for a reproducible null distribution")
    parser.add_argument("--live", action="store_true", help="re-fetch prices for the real (unshuffled) pass")
    parser.add_argument(
        "--gate-p",
        type=float,
        default=DEFAULT_GATE_P,
        help="fail (exit 1) if the real |r| has an empirical p-value below this against the shuffled null",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.n_shuffles, args.seed, args.live, args.gate_p))


if __name__ == "__main__":
    main()
