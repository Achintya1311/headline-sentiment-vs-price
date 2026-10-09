"""Day 8 CLI: the leakage/audit pass - shuffle headline timestamps and
confirm any sentiment/return signal disappears.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 0

NEXT_STEPS.md's "done when": shuffle the headline timestamps and the signal
must disappear; if a shuffled-timestamp control still predicts returns, the
pipeline is leaking and the result is an artifact.

Running that shuffle on the real fixture alone is not actually a useful
test here, and that is itself worth recording rather than hiding: Day 5/6
already found the real contemporaneous/lagged correlation is statistically
indistinguishable from zero (r=-0.185, 95% CI [-0.555, +0.246], n=23).
Shuffling timestamps on data that already shows no signal will *also* show
no signal, whether or not the alignment pipeline has a leak - a no-op
shuffle control passes trivially on null data, so it has no power to catch
anything.

So this module runs two checks, and the real NEXT_STEPS.md gate is the
second one, made meaningful by the first:

1. ``positive_control_headlines`` builds a synthetic headline set from this
   repo's own committed price fixtures: for each (ticker, trading day) with
   a non-trivial open-to-close return, one headline is generated whose text
   is strongly positive-sentiment if that day's return was positive, and
   strongly negative if it was negative - a genuine signal, by
   construction, correctly aligned (pre-open local time on that exact
   session). These headlines are synthetic test fixtures for this audit
   only; they are never real news and the module says so everywhere they
   appear.
2. The real test: shuffle ``published_at`` across headlines - titles (and
   therefore VADER compound scores) stay with their original row; only the
   timestamp that drives ``session_date`` moves. On the positive control,
   this severs the engineered sentiment/return pairing, so the correlation
   should collapse toward zero. A permutation test (many shuffles) checks
   that the real-pairing correlation is a clear outlier against the
   shuffled-pairing null distribution - proof the shuffle control actually
   has the power to detect a real signal collapsing, not just that it
   passes vacuously.

The same shuffle is then run on the real fixture, for the literal
NEXT_STEPS.md gate - with the caveat from above recorded plainly in the
CLI's own output and in the README, not left for a reader to discover.

Exit code is non-zero if either check fails: the positive control not
showing a strong real-pairing correlation (the synthetic construction
itself is broken), the permutation test not showing that correlation
collapse under shuffling (the audit has no power), or the real fixture's
shuffled distribution looking suspiciously different from its own
real-pairing result (a possible leak on the data that matters).
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.prices import PriceFetchError, load_bars
from sentiment.stats import pearson_with_ci

IST = ZoneInfo("Asia/Kolkata")

# (ticker, up-title, down-title). Title text is deliberately extreme and
# deliberately fake-sounding - these are audit-only synthetic fixtures, not
# claims about any real company, and VADER's compound score for each (see
# tests/test_audit.py) is large and reliably signed, which is the only
# property this control needs.
_POSITIVE_CONTROL_TEMPLATES: list[tuple[str, str, str]] = [
    (
        "FORTIS.NS",
        "Fortis Healthcare shares surge as fantastic outstanding excellent results beat all expectations",
        "Fortis Healthcare shares plunge as disastrous terrible horrendous results miss all expectations",
    ),
    (
        "GESHIP.NS",
        "Great Eastern Shipping shares soar on wonderful fantastic record profit surge",
        "Great Eastern Shipping shares crash on terrible dreadful massive loss collapse",
    ),
    (
        "MFSL.NS",
        "Jefferies names Max Financial a fantastic wonderful excellent top pick today",
        "Jefferies names Max Financial a terrible dreadful awful worst pick today",
    ),
    (
        "HDFCBANK.NS",
        "HDFC Bank shares jump on fantastic outstanding wonderful earnings beat",
        "HDFC Bank shares tumble on terrible dreadful disastrous earnings miss",
    ),
    (
        "SUZLON.NS",
        "Suzlon Energy shares rally on fantastic wonderful excellent order win",
        "Suzlon Energy shares slump on terrible dreadful horrible order loss",
    ),
    (
        "BSE.NS",
        "BSE shares climb on fantastic wonderful excellent volume surge",
        "BSE shares sink on terrible dreadful awful volume collapse",
    ),
]

# Below this magnitude a session's return is treated as flat (several
# fixture tickers carry an exact 0.0 return on the same date - almost
# certainly a holiday placeholder bar, not a real signed move) and dropped:
# a flat day gives the construction no sign to encode.
FLAT_RETURN_EPS = 1e-6

SCRAPED_AT = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)


def positive_control_headlines(live: bool = False) -> list[Headline]:
    """Build the synthetic, by-construction-correlated headline set described
    in this module's docstring. Raises ``PriceFetchError`` if a template's
    fixture is missing - this control depends on fixtures already committed
    for Day 5's real tickers, so that should never happen offline."""
    headlines: list[Headline] = []
    for ticker, up_title, down_title in _POSITIVE_CONTROL_TEMPLATES:
        bars = load_bars(ticker, live=live)
        for bar in bars:
            ret = bar.session_return
            if abs(ret) < FLAT_RETURN_EPS:
                continue
            title = up_title if ret > 0 else down_title
            published_at = datetime.combine(bar.date, datetime.min.time(), tzinfo=IST).replace(hour=8)
            headlines.append(
                Headline(
                    source="synthetic-audit",
                    title=title,
                    link=f"synthetic-audit://{ticker}/{bar.date.isoformat()}/{'up' if ret > 0 else 'down'}",
                    published_at=published_at,
                    published_raw="synthetic, generated by sentiment.audit",
                    scraped_at=SCRAPED_AT,
                )
            )
    return headlines


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with the same headlines but ``published_at`` values
    permuted across them - breaks any genuine timestamp-to-content
    relationship while leaving every title (and therefore every VADER
    score) exactly where it was, which is what "shuffle the headline
    timestamps" means: the leak this guards against is in the
    timestamp-to-session pairing, not in the scorer."""
    timestamps = [h.published_at for h in headlines]
    shuffled = timestamps[:]
    rng.shuffle(shuffled)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled)]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> float | None:
    """Pearson r between VADER compound and contemporaneous return across
    ``headlines``, or ``None`` if fewer than 2 resolve to a priced ticker."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_with_ci(compounds, returns).r


@dataclass(frozen=True)
class PermutationResult:
    real_r: float
    n_shuffles: int
    shuffled_rs: list[float]
    # Two-sided permutation p-value: the fraction of shuffles whose |r| is
    # at least as large as the real-pairing |r|. Small means the real
    # pairing is an outlier against the shuffled null - what a genuine,
    # non-leaked signal should look like.
    p_value: float

    @property
    def mean_abs_shuffled_r(self) -> float:
        return sum(abs(r) for r in self.shuffled_rs) / len(self.shuffled_rs)


def permutation_test(
    headlines: list[Headline], n_shuffles: int, seed: int, live: bool = False
) -> PermutationResult:
    real_r = contemporaneous_r(headlines, live=live)
    if real_r is None:
        raise ValueError("fewer than 2 headlines resolved to a priced ticker - nothing to test")

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_published_at(headlines, rng)
        r = contemporaneous_r(shuffled, live=live)
        shuffled_rs.append(r if r is not None else 0.0)

    hits = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    p_value = hits / n_shuffles
    return PermutationResult(real_r=real_r, n_shuffles=n_shuffles, shuffled_rs=shuffled_rs, p_value=p_value)


# Thresholds the CLI's exit code is judged against. These are deliberately
# loose - this is a sanity gate on the audit's own power, not a tuned
# statistical test - see README Findings for the actual numbers observed.
CONTROL_MIN_ABS_R = 0.5
CONTROL_MAX_P_VALUE = 0.05


def run(n_shuffles: int, seed: int, live: bool) -> int:
    ok = True

    print("== positive control: synthetic signal, engineered by construction ==")
    try:
        control_headlines = positive_control_headlines(live=live)
    except PriceFetchError as exc:
        print(f"could not build the positive control: {exc}", file=sys.stderr)
        return 1

    control = permutation_test(control_headlines, n_shuffles=n_shuffles, seed=seed, live=live)
    print(f"{len(control_headlines)} synthetic headlines, real pairing: r={control.real_r:+.3f}")
    print(
        f"after {control.n_shuffles} timestamp shuffles: mean |r|={control.mean_abs_shuffled_r:.3f}  "
        f"permutation p={control.p_value:.4f}"
    )
    if abs(control.real_r) < CONTROL_MIN_ABS_R:
        print(
            f"FAIL: real-pairing |r|={abs(control.real_r):.3f} is below {CONTROL_MIN_ABS_R} - the "
            "synthetic construction itself isn't producing a signal, so this audit can't prove anything.",
            file=sys.stderr,
        )
        ok = False
    elif control.p_value > CONTROL_MAX_P_VALUE:
        print(
            f"FAIL: permutation p={control.p_value:.4f} is above {CONTROL_MAX_P_VALUE} - shuffling "
            "timestamps did not make the engineered signal collapse. The audit has no power to catch a leak.",
            file=sys.stderr,
        )
        ok = False
    else:
        print("PASS: the engineered signal is real under the correct pairing and collapses under shuffling.")

    print()
    print("== real fixture: the literal NEXT_STEPS.md gate ==")
    real_headlines = read_csv(DEFAULT_IN)
    real = permutation_test(real_headlines, n_shuffles=n_shuffles, seed=seed, live=live)
    print(f"{len(real_headlines)} real headlines, real pairing: r={real.real_r:+.3f}")
    print(
        f"after {real.n_shuffles} timestamp shuffles: mean |r|={real.mean_abs_shuffled_r:.3f}  "
        f"permutation p={real.p_value:.4f}"
    )
    print(
        "Day 5/6 already found this real-pairing r is statistically indistinguishable from zero "
        "(95% CI contains 0, n=23) - so this check mostly confirms null stays null, not that a real "
        "signal collapses. It has little power to catch a leak on this fixture; the positive control "
        "above is what actually demonstrates the audit works. See README Limitations."
    )
    if real.p_value <= CONTROL_MAX_P_VALUE:
        print(
            f"FAIL: real-pairing r={real.real_r:+.3f} is an outlier against its own shuffled null "
            f"(p={real.p_value:.4f}) despite Day 5/6's own finding that this correlation's CI contains "
            "zero - that combination is the leak NEXT_STEPS.md warns about and should be investigated "
            "before trusting any number from this pipeline.",
            file=sys.stderr,
        )
        ok = False
    else:
        print("PASS: no leak detected - the real result is as consistent with the shuffled null as expected.")

    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-shuffles", type=int, default=500, help="number of timestamp-shuffle replicates")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for a deterministic report")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
