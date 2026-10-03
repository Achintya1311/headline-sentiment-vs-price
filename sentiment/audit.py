"""Day 8 CLI: the ml-pipeline-audit leakage control.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 1
    python -m sentiment.audit --live

The repo's own "Done when" bar (NEXT_STEPS.md, README Correctness gate):
shuffle the headline timestamps and the signal must disappear. If a
shuffled-timestamp control still predicts returns as well as the real
timestamps do, the pipeline is leaking something that does not actually
depend on when a headline was published - order, or some other accidental
correlation - and Day 5's correlation cannot be trusted.

This redistributes each headline's ``published_at`` across the other
headlines in the same fixture, keeping every headline's own title (and so
its VADER score) fixed. Re-running Day 4's alignment and Day 5's
contemporaneous-return pairing on the shuffled copy re-pairs each
headline's sentiment with whichever session its *reassigned* timestamp
now lands on - a pairing that, if the real pipeline were leak-free, should
carry no more information than chance.

This fixture's real correlation is already a null result (Day 5 README
Findings: r=-0.185, 95% CI comfortably containing zero), so this CLI's own
numbers can only show the shuffle test fails to manufacture a false
positive out of this fixture - it cannot demonstrate the control has the
power to detect and destroy a *real* signal, because there is no real
signal here to destroy. ``tests/test_audit.py`` has the test that
actually proves that, against a synthetic dataset built to contain one.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import PriceFetchError, bar_on, load_bars
from sentiment.stats import PearsonResult, pearson_with_ci
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_N_SHUFFLES = 1000


def build_rows(headlines: list[Headline], live: bool = False) -> list[dict]:
    """The same resolve -> align -> price-bar -> score pairing
    ``sentiment.correlate.build_rows`` does, but over an explicit headline
    list rather than a CSV path, so a shuffled-timestamp copy can be paired
    without touching disk. Unlike ``correlate.build_rows`` this drops
    unresolved/price-fetch failures silently - the audit's point is the
    distribution of correlations across many shuffles, not re-reporting the
    per-headline accounting ``sentiment.correlate`` already does once.
    """
    rows: list[dict] = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _, ticker = match
        if ticker is None:
            continue

        alignment = align_headline(h.published_at)
        try:
            bars = load_bars(ticker, live=live)
        except PriceFetchError:
            continue
        session_bar = bar_on(bars, alignment.session_date)
        if session_bar is None:
            continue

        scored = score_headline(h)
        rows.append({"compound": scored.compound, "contemporaneous_return": session_bar.session_return})
    return rows


def correlation(headlines: list[Headline], live: bool = False) -> PearsonResult | None:
    rows = build_rows(headlines, live=live)
    if len(rows) < 2:
        return None
    return pearson_with_ci([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Redistribute ``published_at`` across ``headlines``, keeping every
    headline's own content (source/title/link, and so its sentiment score)
    fixed. A headline that only "predicts" its paired session's return
    because of its real timing - not because of what it says - should lose
    that pairing once timestamps are randomly reassigned among the same set
    of headlines.
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


def shuffle_null_distribution(
    headlines: list[Headline], n_shuffles: int, seed: int, live: bool = False
) -> list[float]:
    """``n_shuffles`` independent timestamp-shuffle correlations (``|r|``).

    A shuffle that happens to leave fewer than 2 resolvable rows is dropped
    rather than counted as zero - ``resolve()`` and the price fetch do not
    depend on ``published_at``, so this is rare, and dropping it avoids
    pulling the null distribution toward zero for a reason that has nothing
    to do with leakage.
    """
    rng = random.Random(seed)
    rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        result = correlation(shuffled, live=live)
        if result is not None:
            rs.append(abs(result.r))
    return rs


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    real = correlation(headlines, live=live)
    if real is None:
        print("not enough resolved headlines for a correlation; nothing to audit", file=sys.stderr)
        return 1

    print(
        f"real (unshuffled) contemporaneous: r={real.r:+.3f}  "
        f"95% CI [{real.ci_low:+.3f}, {real.ci_high:+.3f}]  n={real.n}"
    )

    null_rs = shuffle_null_distribution(headlines, n_shuffles=n_shuffles, seed=seed, live=live)
    if not null_rs:
        print("every shuffle left fewer than 2 resolvable rows; cannot build a null distribution", file=sys.stderr)
        return 1

    mean_null = sum(null_rs) / len(null_rs)
    exceed = sum(1 for r in null_rs if r >= abs(real.r))
    p_value = exceed / len(null_rs)

    print(f"shuffled-timestamp null (n={len(null_rs)} shuffles): mean |r|={mean_null:.3f}")
    print(f"empirical p-value (shuffled |r| >= real |r|): {p_value:.3f}")

    if abs(real.r) <= 2 * mean_null:
        print(
            "verdict: real |r| sits inside the range chance timestamp-shuffling produces on its own - "
            "no evidence this correlation depends on anything the pipeline is leaking. Consistent with "
            "Day 5's own finding that this fixture shows no signal to begin with."
        )
    else:
        print(
            "verdict: WARNING - real |r| is well above what shuffled timestamps produce by chance. "
            "Investigate for a look-ahead leak before trusting this correlation."
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle trials"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the shuffle RNG (deterministic by default)")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
