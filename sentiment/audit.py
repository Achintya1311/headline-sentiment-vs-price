"""Day 8 CLI: ml-pipeline-audit - the leakage control NEXT_STEPS.md and the
README's "Done when" section commit to before any correlation result here
is believed.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000
    python -m sentiment.audit --event-threshold 0.3

The idea: shuffle which ``published_at`` timestamp belongs to which
headline, keeping everything else (title, source, scored sentiment, which
ticker it resolves to) fixed, then re-run the *same* Day 4/5 alignment and
pairing logic (``sentiment.market_hours.align_headline`` by way of
``sentiment.correlate.rows_from_headlines``) on the shuffled corpus. A
shuffled corpus has no honest reason to predict returns - any apparent
correlation it still shows would mean the alignment pipeline is leaking
information some other way, not through genuine timing. Doing this many
times builds a null distribution to compare the real, unshuffled
correlation against.

Two checks, not one:

- ``audit_real_fixture`` runs the shuffle against this repo's actual 50
  scraped headlines. Given Day 5's own null finding (r=-0.185, CI crossing
  zero), the expected, *passing* result is unremarkable: the real r sits
  comfortably inside the shuffled null band. That is a necessary check, but
  on its own it is a weak one - a test with no power to detect a real leak
  would also report "no difference found."
- ``audit_synthetic_leak`` is the positive control that gives the first
  check teeth: it builds a deliberately leaky corpus (sentiment wording
  chosen, headline by headline, to match the *sign* of that headline's own
  ticker's same-session return, something a real scraper has no way to do
  honestly) and runs the identical shuffle procedure against it. If the
  leaky corpus's real correlation does *not* collapse once timestamps are
  shuffled, this audit methodology has no power to catch a real leak and
  the first check above is not evidence of anything.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0
MIN_ROWS_FOR_R = 2


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float | None
    real_n: int
    shuffled_rs: list[float]
    p_value: float | None  # two-sided permutation p-value; None if undefined

    @property
    def shuffled_mean(self) -> float | None:
        if not self.shuffled_rs:
            return None
        return sum(self.shuffled_rs) / len(self.shuffled_rs)


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Same headlines, same corpus of ``published_at`` values, but randomly
    reassigned across headlines - same content and sentiment, wrong clock.

    This is the control: it breaks whatever real relationship exists between
    a headline's actual timing and the session Day 4's alignment assigns it,
    while leaving title (hence sentiment score and ticker resolution)
    untouched.
    """
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


def _correlate_rows(rows: list[dict]) -> tuple[float | None, int]:
    """Pearson r between ``compound`` and ``contemporaneous_return`` across
    ``rows``, or ``(None, n)`` if there are too few rows to define one."""
    if len(rows) < MIN_ROWS_FOR_R:
        return None, len(rows)
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), len(rows)


def run_shuffle_audit(
    headlines: list[Headline],
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> ShuffleAuditResult:
    """Pair ``headlines`` against price data as-is, then do the same ``n_shuffles``
    times with timestamps shuffled, and report a permutation p-value for how
    unusual the real correlation is against that shuffled null distribution."""
    real_rows, _ = rows_from_headlines(headlines, live=live)
    real_r, real_n = _correlate_rows(real_rows)

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled_headlines = shuffle_timestamps(headlines, rng)
        shuffled_rows, _ = rows_from_headlines(shuffled_headlines, live=live)
        r, n = _correlate_rows(shuffled_rows)
        if r is not None:
            shuffled_rs.append(r)

    p_value = None
    if real_r is not None and shuffled_rs:
        extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
        # +1/+1 (add-one smoothing): a permutation test can never honestly
        # report p=0 from a finite sample - the real draw is itself one of
        # the possible outcomes under the null.
        p_value = (extreme + 1) / (len(shuffled_rs) + 1)

    return ShuffleAuditResult(real_r=real_r, real_n=real_n, shuffled_rs=shuffled_rs, p_value=p_value)


def audit_real_fixture(in_path: Path = DEFAULT_IN, n_shuffles: int = DEFAULT_N_SHUFFLES, seed: int = DEFAULT_SEED) -> ShuffleAuditResult:
    """The audit this repo's own result has to pass: shuffle the real scraped
    headlines' timestamps and confirm the real correlation is not an outlier
    against the shuffled null - i.e. nothing about the alignment pipeline is
    manufacturing a correlation that a random clock would not also produce."""
    headlines = read_csv(in_path)
    return run_shuffle_audit(headlines, n_shuffles=n_shuffles, seed=seed)


# --- synthetic positive control ---------------------------------------

# (ticker, session_date, sign) - ticker/date pairs picked from this repo's
# own committed fixtures/prices/*.json for a visible same-session return in
# the given direction (see the day's checkpoint note for how these were
# chosen - top absolute-return days across the universe Day 5 already
# resolves tickers against). The leak: wording is chosen to match ``sign``,
# which a real headline scraper has no honest way to do, since it would
# require already knowing that session's own close.
_LEAK_TICKER_DATES: list[tuple[str, str, str]] = [
    ("POLICYBZR.NS", "2026-09-24", "neg"),
    ("PCJEWELLER.NS", "2026-09-07", "pos"),
    ("PCJEWELLER.NS", "2026-09-04", "pos"),
    ("PCJEWELLER.NS", "2026-09-28", "neg"),
    ("POLICYBZR.NS", "2026-09-29", "neg"),
    ("PCJEWELLER.NS", "2026-09-15", "neg"),
    ("PCJEWELLER.NS", "2026-09-22", "pos"),
    ("PCJEWELLER.NS", "2026-09-01", "neg"),
    ("NESTLEIND.NS", "2026-09-01", "neg"),
    ("POLICYBZR.NS", "2026-09-16", "pos"),
    ("MFSL.NS", "2026-09-17", "pos"),
    ("PCJEWELLER.NS", "2026-09-08", "neg"),
    ("MFSL.NS", "2026-09-25", "pos"),
    ("PCJEWELLER.NS", "2026-09-23", "pos"),
    ("POLICYBZR.NS", "2026-09-23", "pos"),
    ("MFSL.NS", "2026-09-29", "neg"),
    ("FORTIS.NS", "2026-09-25", "neg"),
    ("HDFCLIFE.NS", "2026-09-17", "pos"),
    ("SBILIFE.NS", "2026-09-17", "pos"),
    ("MFSL.NS", "2026-09-16", "pos"),
]

# Ticker -> a title prefix tickers.py's resolve() table maps back to that
# same ticker (see sentiment/tickers.py's _RULES).
_TICKER_TITLE_PREFIX: dict[str, str] = {
    "POLICYBZR.NS": "PB Fintech shares",
    "PCJEWELLER.NS": "PC Jeweller shares",
    "NESTLEIND.NS": "Nestle India Share Price Highlights:",
    "MFSL.NS": "Jefferies names Max Financial a top pick -",
    "FORTIS.NS": "Fortis Healthcare shares",
    "HDFCLIFE.NS": "HDFC Life Share Price Highlights:",
    "SBILIFE.NS": "SBI Life Share Price Highlights:",
}

_POSITIVE_WORDING = "surge on absolutely excellent, outstanding, fantastic, wonderful, amazing results"
_NEGATIVE_WORDING = "plunge on absolutely disastrous, terrible, horrible, awful, dreadful results"

# 09:00 IST (03:30 UTC) is before the 09:15 IST open, so every synthetic
# headline is pre-open and aligns to the same calendar date it names -
# the leak is in the wording, not in exercising Day 4's roll-forward cases.
_PRE_OPEN_UTC_TIME = "03:30:00"


def build_synthetic_leak_headlines() -> list[Headline]:
    """A headline corpus with a deliberate, constructed leak: each
    headline's wording matches the *sign* of its own ticker's same-session
    return (see ``_LEAK_TICKER_DATES``). Real reporting cannot do this - it
    would require already knowing the session's own close - so if the
    shuffle audit cannot flag this corpus, it cannot be trusted to flag a
    real one either."""
    headlines = []
    for i, (ticker, session_date, sign) in enumerate(_LEAK_TICKER_DATES):
        prefix = _TICKER_TITLE_PREFIX[ticker]
        wording = _POSITIVE_WORDING if sign == "pos" else _NEGATIVE_WORDING
        title = f"{prefix} {ticker.split('.')[0]} shares {wording}"
        published_at = datetime.fromisoformat(f"{session_date}T{_PRE_OPEN_UTC_TIME}+00:00").astimezone(timezone.utc)
        headlines.append(
            Headline(
                source="synthetic_leak_control",
                title=title,
                link=f"https://example.invalid/synthetic-leak/{i}",
                published_at=published_at,
                published_raw=published_at.isoformat(),
                scraped_at=published_at,
            )
        )
    return headlines


def audit_synthetic_leak(n_shuffles: int = DEFAULT_N_SHUFFLES, seed: int = DEFAULT_SEED) -> ShuffleAuditResult:
    """The positive control: run the same shuffle procedure against a corpus
    built to leak by construction, and confirm the real (unshuffled)
    correlation is a clear outlier against the shuffled null - i.e. the
    audit methodology actually has the power to catch a leak when one
    exists, not just a no-op that always says "fine"."""
    return run_shuffle_audit(build_synthetic_leak_headlines(), n_shuffles=n_shuffles, seed=seed)


def _print_result(label: str, result: ShuffleAuditResult) -> None:
    if result.real_r is None:
        print(f"{label}: not enough resolved headlines for a correlation (n={result.real_n})")
        return
    print(f"{label}: real r={result.real_r:+.3f}  n={result.real_n}")
    if result.shuffled_rs:
        print(
            f"  shuffled null (n={len(result.shuffled_rs)} shuffles): "
            f"mean r={result.shuffled_mean:+.3f}  "
            f"range [{min(result.shuffled_rs):+.3f}, {max(result.shuffled_rs):+.3f}]"
        )
    if result.p_value is not None:
        print(f"  permutation p-value (two-sided): {result.p_value:.4f}")


def run(n_shuffles: int, seed: int, in_path: Path) -> int:
    print("Real fixture (this repo's own 50 scraped headlines) - the leakage gate:")
    real_audit = audit_real_fixture(in_path, n_shuffles=n_shuffles, seed=seed)
    _print_result("  real fixture", real_audit)
    if real_audit.p_value is not None:
        if real_audit.p_value < 0.05:
            print(
                "  FLAG: the real correlation is an outlier against its own shuffled-timestamp "
                "null (p < 0.05) - investigate before trusting this result."
            )
        else:
            print(
                "  no flag: the real correlation is not distinguishable from a shuffled-timestamp "
                "control (p >= 0.05) - consistent with Day 5's own null finding, not evidence that "
                "a real signal would also disappear this way."
            )

    print()
    print("Synthetic leak control (deliberately leaky corpus) - proves the audit has power:")
    leak_audit = audit_synthetic_leak(n_shuffles=n_shuffles, seed=seed)
    _print_result("  synthetic leak", leak_audit)
    if leak_audit.p_value is not None:
        if leak_audit.p_value < 0.05:
            print("  as expected: the constructed leak is flagged (p < 0.05) - the shuffle has power.")
        else:
            print(
                "  WARNING: the constructed leak was NOT flagged - this audit methodology has no "
                "demonstrated power to catch a real leak; the real-fixture result above is not "
                "evidence of anything."
            )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp shuffles")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic null")
    args = parser.parse_args()
    sys.exit(run(args.n_shuffles, args.seed, args.in_path))


if __name__ == "__main__":
    main()
