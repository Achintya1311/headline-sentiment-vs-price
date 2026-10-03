"""Day 8 CLI: the timestamp-shuffle leakage audit NEXT_STEPS.md calls "done when".

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000 --seed 1
    python -m sentiment.audit --live

The correctness gate for this whole repo, per the README: randomise headline
publish times and the contemporaneous sentiment/return correlation must not
come out looking any stronger than chance. If a shuffled-timestamp control
still "predicts" returns, something upstream of the correlation is leaking -
most plausibly ``sentiment.market_hours.align_headline`` letting a return
that predates the headline into the pairing, the exact look-ahead class of
bug Day 4 exists to prevent.

This only shuffles ``published_at`` across headlines - title, source, link
and (downstream) the VADER compound and the resolved ticker all stay with
the headline they belong to. What gets severed is the link between *what a
headline says* and *when it was actually published*, which is the one thing
a look-ahead leak could be exploiting. The multiset of timestamps is exactly
preserved (a permutation, not a resample), so a shuffle can never invent a
trading session that was not already reachable by some real headline.

What this test can and cannot show, given Day 5-7's own honest findings:
Day 5's real contemporaneous correlation is already a null result
(r=-0.185, CI comfortably straddling zero). A leakage test built to check
that "a real signal disappears under shuffling" has nothing to make
disappear here - there was no signal to begin with. What it *can* still
check is the weaker but real claim this repo can actually stand behind: the
real-timestamp result is not a statistical outlier against what pure chance
plus a random headline-to-session pairing already produces. See the
README's Day 8 Findings and "why this might be spurious" section for what
that does and does not rule out.
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
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0

# How extreme the real result is allowed to be against the shuffled null
# before this is flagged as a possible leak. 0.95 means: the real |r| must
# not sit above the 95th percentile of shuffled |r|'s.
DEFAULT_PASS_PERCENTILE = 0.95


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with ``published_at`` permuted across ``headlines``.

    Every other field stays with its original headline - only the pairing
    between a headline's content and its publish time is broken, and the
    exact multiset of timestamps is preserved (``rng.shuffle`` on a copy),
    so this can only reassign headlines to sessions that genuinely exist in
    this data, never invent one.
    """
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, ts in zip(headlines, timestamps)
    ]


def contemporaneous_r(in_path: Path, live: bool = False) -> tuple[float, int]:
    """Pearson r between VADER compound and contemporaneous return for every
    headline ``build_rows`` resolves from ``in_path``, plus how many rows
    that was. ``r=0.0`` for fewer than 2 resolved rows, the same degrade-
    gracefully convention ``sentiment.correlate.run`` already uses."""
    rows, _ = build_rows(in_path, live=live)
    if len(rows) < 2:
        return 0.0, len(rows)
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), len(rows)


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]

    @property
    def n_shuffles(self) -> int:
        return len(self.shuffled_rs)

    @property
    def shuffled_mean(self) -> float:
        return sum(self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def shuffled_std(self) -> float:
        mean = self.shuffled_mean
        var = sum((r - mean) ** 2 for r in self.shuffled_rs) / len(self.shuffled_rs)
        return var**0.5

    def percentile_of_real(self) -> float:
        """Fraction of shuffled |r|'s at or below the real result's |r|.

        This is the actual check: not "is the real r close to zero" (a
        genuine, un-leaked effect need not be), but "is the real r bigger
        than what randomly reassigning timestamps already produces by
        chance." A value near 1.0 means the real alignment finds a
        relationship shuffled alignments essentially never do.
        """
        real_abs = abs(self.real_r)
        at_or_below = sum(1 for r in self.shuffled_rs if abs(r) <= real_abs)
        return at_or_below / len(self.shuffled_rs)

    def passes(self, percentile: float = DEFAULT_PASS_PERCENTILE) -> bool:
        """True if the real result is not an extreme outlier against the
        shuffled null (see ``percentile_of_real``). False is the leakage
        signature this test exists to catch - see the module docstring for
        what it means (and does not mean) when the real result is already
        a null finding on its own."""
        return self.percentile_of_real() < percentile


def run_audit(
    in_path: Path,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> AuditResult:
    real_r, real_n = contemporaneous_r(in_path, live=live)

    headlines = read_csv(in_path)
    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "shuffled_headlines.csv"
        for _ in range(n_shuffles):
            shuffled = shuffle_published_at(headlines, rng)
            write_csv(shuffled, tmp_path)
            r, n = contemporaneous_r(tmp_path, live=live)
            if n >= 2:
                shuffled_rs.append(r)

    if not shuffled_rs:
        raise RuntimeError(
            "every shuffle trial resolved fewer than 2 rows; nothing to compare the real "
            "result against - try more shuffles or check the price fixtures cover a wider "
            "date range"
        )

    return AuditResult(real_r=real_r, real_n=real_n, shuffled_rs=shuffled_rs)


def run(
    in_path: Path,
    n_shuffles: int,
    seed: int,
    live: bool,
    percentile: float = DEFAULT_PASS_PERCENTILE,
) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = run_audit(in_path, n_shuffles=n_shuffles, seed=seed, live=live)

    print(f"real (correct timestamps):  r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled null ({result.n_shuffles} trials): "
        f"mean r={result.shuffled_mean:+.3f}  std={result.shuffled_std:.3f}"
    )
    print(
        f"real |r| is at the {result.percentile_of_real():.1%} percentile of the shuffled |r| distribution"
    )

    if result.passes(percentile):
        print(f"PASS: real result is not an outlier against the shuffled null (threshold {percentile:.0%})")
        return 0
    print(f"FAIL: real result exceeds {percentile:.0%} of the shuffled null - possible leakage, investigate")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of shuffle trials")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the shuffle RNG (reproducible)")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PASS_PERCENTILE,
        help="flag the real result if its |r| exceeds this fraction of the shuffled null",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live, args.percentile))


if __name__ == "__main__":
    main()
