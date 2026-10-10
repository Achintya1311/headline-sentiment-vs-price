"""Day 8 CLI: the leakage audit (shuffle headline timestamps, confirm the
sentiment/return signal disappears) plus the pass/fail call NEXT_STEPS.md's
"Done when" names.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 1

The control: take the real, correctly-aligned (headline, lagged return)
pairing Day 5/6 already build, then permute *only* the timestamps across the
same set of headlines (same titles, same VADER compound scores, same set of
published_at values - just reassigned to different headlines) and rebuild
the pairing from scratch through the real ``sentiment.market_hours``
alignment and ``sentiment.prices`` lookup. Content and sentiment never
move; only which trading session each headline is credited with reacting to
does.

If the real pairing's correlation is unremarkable next to the distribution
of correlations the shuffled control produces, that is what "no leakage"
looks like: the (possibly weak or null) signal is not being manufactured by
something other than correct timestamp alignment. If a shuffled run kept
producing a correlation as strong as the real one, that would mean the
pipeline's signal does not actually depend on getting timestamps right -
exactly the leak NEXT_STEPS.md's "Done when" is checking for.

See the README's Day 8 Findings for why this fixture's real result is
already a near-null finding (Day 5/6 already found that), and
``tests/test_audit.py`` for a synthetic positive control proving this audit
would catch a real signal collapsing under shuffling, not just rubber-stamp
a dataset that has nothing to leak.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows
from sentiment.headline import Headline, read_csv, write_csv
from sentiment.stats import pearson_with_ci

DEFAULT_N_SHUFFLES = 200
DEFAULT_SEED = 0
MIN_ROWS_FOR_CORRELATION = 2


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Same headlines, same set of ``published_at`` values - permuted onto
    different headlines. This is the "shuffled-timestamp control" NEXT_STEPS.md
    names: it breaks the link between a headline's real publish time (and so
    the trading session Day 4's alignment assigns it) and its own title/score,
    without changing the marginal mix of pre-open/intraday/post-close timing
    the real data has.
    """
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=new_ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, new_ts in zip(headlines, shuffled_timestamps)
    ]


def lagged_correlation(rows: list[dict]) -> tuple[float, int]:
    """Pearson r between ``compound`` and ``lagged_return`` over the rows
    that have a lagged return at all. Returns ``(r, n)``; ``r`` is reported
    as ``0.0`` when ``n`` is too small for a correlation to mean anything -
    the caller decides what to do with a small ``n``, this never raises."""
    lagged = [r for r in rows if r["lagged_return"] is not None]
    if len(lagged) < MIN_ROWS_FOR_CORRELATION:
        return 0.0, len(lagged)
    compounds = [r["compound"] for r in lagged]
    returns = [r["lagged_return"] for r in lagged]
    return pearson_with_ci(compounds, returns).r, len(lagged)


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]

    @property
    def n_shuffles_used(self) -> int:
        """Shuffles that actually produced a correlation (n >= 2 lagged
        rows survived the shuffle - see README Limitations on why a shuffle
        can occasionally leave too few)."""
        return len(self.shuffled_rs)

    @property
    def shuffled_mean(self) -> float:
        return sum(self.shuffled_rs) / len(self.shuffled_rs) if self.shuffled_rs else 0.0

    @property
    def permutation_p_value(self) -> float:
        """Two-sided permutation-test p-value: the fraction of shuffles whose
        |r| meets or beats the real |r|. High (close to 1) means the real
        statistic is unremarkable against the scrambled-timestamp null -
        the passing case. Low means the real statistic stands out even
        after scrambling, which given this audit shuffles away the one
        thing (correct alignment) a genuine signal depends on, would point
        at leakage rather than a real effect."""
        if not self.shuffled_rs:
            return 1.0
        at_least_as_extreme = sum(1 for r in self.shuffled_rs if abs(r) >= abs(self.real_r))
        return at_least_as_extreme / len(self.shuffled_rs)

    @property
    def passes(self) -> bool:
        """The leakage test NEXT_STEPS.md's "Done when" names: the real
        result must not be an outlier against the shuffled null. A 5%
        two-sided permutation threshold is the usual convention."""
        return self.permutation_p_value >= 0.05


def run_audit(in_path: Path, live: bool, n_shuffles: int, seed: int) -> AuditResult:
    real_rows, _ = build_rows(in_path, live=live)
    real_r, real_n = lagged_correlation(real_rows)

    headlines = read_csv(in_path)
    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "shuffled_headlines.csv"
        for _ in range(n_shuffles):
            shuffled = shuffle_timestamps(headlines, rng)
            write_csv(shuffled, tmp_path)
            shuffled_rows, _ = build_rows(tmp_path, live=live)
            r, n = lagged_correlation(shuffled_rows)
            if n >= MIN_ROWS_FOR_CORRELATION:
                shuffled_rs.append(r)

    return AuditResult(real_r=real_r, real_n=real_n, shuffled_rs=shuffled_rs)


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = run_audit(in_path, live, n_shuffles, seed)

    if result.real_n < MIN_ROWS_FOR_CORRELATION:
        print(
            f"real data: only {result.real_n} headline(s) have a lagged return - "
            "not enough to correlate, so there is no signal for this audit to test",
            file=sys.stderr,
        )
        return 1

    print(f"real (correctly aligned):  r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled-timestamp control: mean r={result.shuffled_mean:+.3f}  "
        f"over {result.n_shuffles_used}/{n_shuffles} shuffles"
    )
    print(f"permutation p-value (real r vs shuffled null): {result.permutation_p_value:.3f}")
    if result.passes:
        print("PASS: the real result is not an outlier against the shuffled-timestamp null.")
    else:
        print(
            "FAIL: the real result stands out even after shuffling away timestamp alignment - "
            "see README Day 8 Findings before trusting any other day's correlation number."
        )
    return 0 if result.passes else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of shuffled-timestamp control runs"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible report")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
