"""Day 8 CLI: the leakage/audit pass NEXT_STEPS.md's "Done when" section
names as the test that decides whether this repo is finished.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 1

The control: shuffle headline *timestamps* only - keep each headline's title
(and therefore its VADER ``compound`` score and which ticker it resolves to)
exactly as scraped, but redistribute the 50 real ``published_at`` values
across the 50 headlines at random. Re-run the identical resolve/align/price
pipeline ``sentiment.correlate.build_rows_from_headlines`` already uses, and
recompute the contemporaneous correlation. Repeat many times to build a null
distribution, then ask: is the real (unshuffled) correlation unusual against
that null, or indistinguishable from randomly-timed noise?

Why shuffle timestamps rather than shuffle returns or compounds directly:
timestamps are the one thing Day 4's ``align_headline`` turns into a
session_date, which is what every later day's result actually keys off. A
bug that let a headline's score influence a return it could not honestly
have reacted to (the exact failure mode the "Done when" section names) would
*not* be destroyed by randomising publish time - the leak would live
somewhere else in the pipeline and the shuffled run would keep looking like
the real one. Randomising publish time and watching the correlation change
is therefore a direct probe of whether the result depends on timestamps
being correct, not just a generic shuffle test.

Because this repo's real result is already a null one (see README Day 5/6
Findings), this audit mostly has nothing to disprove - a p-value near 1 here
says "shuffling didn't matter because there was nothing to break," which is
consistent with no leak but is a weak test on its own. ``synthetic_signal_*``
below is the sharper check: it builds a dataset with a *real*, deterministic
sentiment-return relationship wired through ``align_headline`` itself, shows
the unshuffled pipeline recovers it almost exactly, and shows shuffling
destroys it. That is the version of "if a shuffled-timestamp control still
predicts returns, the pipeline is leaking" that actually has teeth - run
against a case where there is something to lose.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 200
DEFAULT_SEED = 0

# How far real sentiment is allowed to move next-day return in the synthetic
# positive control - the exact value doesn't matter, only that it is large
# enough to produce a correlation unmistakably different from noise.
SYNTHETIC_SLOPE = 0.05
SYNTHETIC_N = 40


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Redistribute ``published_at`` across ``headlines`` at random.

    Every other field - source, title, link, published_raw, scraped_at -
    stays with its original headline, so the content a ticker/compound is
    derived from is untouched; only *when* each headline is said to have
    been published changes. ``published_raw`` is left as the original text
    deliberately: it is a record of what the feed actually said, and a
    shuffle run is not a re-scrape.
    """
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


@dataclass(frozen=True)
class PermutationAuditResult:
    real_r: float | None
    real_n: int
    null_rs: list[float]
    p_value: float | None

    @property
    def null_mean(self) -> float:
        return sum(self.null_rs) / len(self.null_rs) if self.null_rs else 0.0

    @property
    def null_std(self) -> float:
        if len(self.null_rs) < 2:
            return 0.0
        m = self.null_mean
        var = sum((r - m) ** 2 for r in self.null_rs) / (len(self.null_rs) - 1)
        return var**0.5


def _contemporaneous_r(headlines: list[Headline], live: bool) -> tuple[float | None, int]:
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None, len(rows)
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), len(rows)


def permutation_audit(
    headlines: list[Headline],
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> PermutationAuditResult:
    """Run the real pipeline once, then ``n_shuffles`` times with timestamps
    shuffled, and report where the real statistic sits in that null
    distribution.

    A shuffled run can resolve to fewer than 2 priced rows (a reassigned
    timestamp can roll a headline's session past the edge of the committed
    one-month price fixture) - those runs are skipped and not counted in
    ``n_shuffles``' denominator, the same "skip, don't fake a number" rule
    ``build_rows`` already applies to unfetchable tickers.
    """
    real_r, real_n = _contemporaneous_r(headlines, live=live)

    rng = random.Random(seed)
    null_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        r, n = _contemporaneous_r(shuffled, live=live)
        if r is not None:
            null_rs.append(r)

    p_value = None
    if real_r is not None and null_rs:
        p_value = sum(1 for r in null_rs if abs(r) >= abs(real_r)) / len(null_rs)

    return PermutationAuditResult(real_r=real_r, real_n=real_n, null_rs=null_rs, p_value=p_value)


def _synthetic_headlines(n: int = SYNTHETIC_N, seed: int = 0) -> tuple[list[Headline], dict[date, float]]:
    """``n`` headlines, one per trading day starting a fixed Monday, each
    published well before that day's 09:15 IST open (so ``align_headline``
    maps headline ``i`` to trading day ``i`` with nothing to roll forward).

    Returns the headlines and ``true_return_by_date``, a day -> return table
    built as a deterministic linear function of each headline's own index -
    standing in for "the market reacted to this sentiment," so there is an
    actual signal here to lose, unlike the real fixture's documented null.
    """
    rng = random.Random(seed)
    day = date(2026, 6, 1)  # a Monday
    headlines: list[Headline] = []
    true_return_by_date: dict[date, float] = {}
    i = 0
    while i < n:
        if day.weekday() < 5:
            compound = rng.uniform(-1.0, 1.0)
            published = datetime.combine(day, time(2, 0), tzinfo=timezone.utc)  # 07:30 IST, pre-open
            headlines.append(
                Headline(
                    source="synthetic",
                    title=f"synthetic headline {i}",
                    link=str(i),
                    published_at=published,
                    published_raw=published.isoformat(),
                    scraped_at=published,
                )
            )
            true_return_by_date[day] = SYNTHETIC_SLOPE * compound
            i += 1
        day += timedelta(days=1)
    return headlines, true_return_by_date


def _synthetic_pairs(headlines: list[Headline], compounds: dict[str, float], true_return_by_date: dict[date, float]):
    """Pair each headline's own ``compound`` with ``true_return_by_date`` at
    the session ``align_headline`` currently assigns it - unshuffled, that is
    the headline's own day; shuffled, it is whichever day the reassigned
    timestamp lands on. A date with no synthetic return (shuffled past the
    edge of the generated range) is dropped, same convention as the real
    pipeline skipping an unpriced session."""
    xs, ys = [], []
    for h in headlines:
        session_date = align_headline(h.published_at).session_date
        if session_date in true_return_by_date:
            xs.append(compounds[h.link])
            ys.append(true_return_by_date[session_date])
    return xs, ys


def synthetic_signal_correlation(shuffle: bool, seed: int = 0) -> float:
    """r between sentiment and return on the synthetic positive-control
    dataset, with or without a timestamp shuffle. Exists so a leak that
    would make a shuffled-timestamp run still predict returns gets caught
    here even on a fixture (the real one) that has no signal to lose."""
    headlines, true_return_by_date = _synthetic_headlines(seed=seed)
    compounds = {h.link: true_return_by_date[align_headline(h.published_at).session_date] / SYNTHETIC_SLOPE for h in headlines}

    rng = random.Random(seed + 1)
    use = shuffle_timestamps(headlines, rng) if shuffle else headlines
    xs, ys = _synthetic_pairs(use, compounds, true_return_by_date)
    return pearson_r(xs, ys)


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = permutation_audit(headlines, n_shuffles=n_shuffles, seed=seed, live=live)

    if result.real_r is not None:
        print(f"real (unshuffled) contemporaneous: r={result.real_r:+.3f}  n={result.real_n}")
    else:
        print(f"real (unshuffled): not enough resolved headlines for a correlation (n={result.real_n})")
    print(
        f"shuffled-timestamp null distribution: {len(result.null_rs)}/{n_shuffles} usable shuffle(s), "
        f"mean r={result.null_mean:+.3f}  std={result.null_std:.3f}"
    )
    if result.p_value is not None:
        print(f"two-sided permutation p-value (|null r| >= |real r|): {result.p_value:.3f}")
        if result.p_value < 0.05:
            print(
                "WARNING: the real correlation is more extreme than 95% of shuffled-timestamp runs. "
                "Either there is a genuine timestamp-dependent effect, or the alignment/price pipeline "
                "is leaking future information past a shuffle that should have destroyed it. Investigate "
                "before trusting this result - see README 'Why this might be spurious'."
            )
        else:
            print(
                "No evidence the real correlation depends on correct timestamps more than chance does - "
                "consistent with Day 5/6's null result, not proof a real effect was missed."
            )
    else:
        print("could not compute a p-value (no usable shuffles or no real correlation)")

    synthetic_unshuffled = synthetic_signal_correlation(shuffle=False, seed=seed)
    synthetic_shuffled = synthetic_signal_correlation(shuffle=True, seed=seed)
    print(
        f"positive control (synthetic, injected signal): unshuffled r={synthetic_unshuffled:+.3f}  "
        f"shuffled r={synthetic_shuffled:+.3f} - shuffling should collapse this towards 0"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp shuffles")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
