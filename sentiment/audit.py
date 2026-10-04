"""Day 8 CLI: the leakage/audit pass the repo's own "Done when" criterion
names - shuffle headline timestamps and confirm the sentiment/return
correlation does not survive.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --n-trials 500

What this actually tests: Day 5's ``sentiment.correlate.build_rows`` pairs
each headline with a trading session via ``sentiment.market_hours.
align_headline(published_at)``. If that alignment step were leaking - if,
say, a bug let a headline see a return that happened before it existed -
then the correlation would depend on the *correct* published_at being paired
with the *correct* headline. Scrambling which published_at goes with which
headline (titles, companies, tickers and compound scores all stay put) and
re-running the identical pipeline breaks that correct pairing. If the
correlation survives the scramble anyway, the result was never actually
coming from the timestamp alignment - it was coming from something else
(a leak, or a coincidence the "real" pairing isn't actually responsible
for), and the whole Day 5 finding would be suspect.

Honest complication, named rather than hidden: Day 5 already found the real
pipeline's correlation is statistically indistinguishable from zero (r=-0.185,
95% CI comfortably containing zero, n=23). A control that is supposed to make
a signal "disappear" cannot demonstrate much by making an already-near-zero
number stay near zero - that would pass even if the shuffle mechanism itself
were broken (e.g. a no-op that forgot to actually shuffle). So this module
runs two checks, not one:

1. **The real check**: shuffle this repo's own fixture many times and show
   the real (unshuffled) r sits inside the shuffled null distribution, not
   out past its tails - consistent with "no detectable signal to leak" per
   Day 5, and the only thing the real data can honestly support.
2. **The positive control** (``tests/test_audit.py``): build a *synthetic*
   headline/price fixture with a real, strong, built-in sentiment/return
   relationship (not this repo's data), confirm the unshuffled pipeline
   detects it, then confirm shuffling destroys it. This is what proves the
   shuffle mechanism itself would have caught a real leak, had this repo's
   fixture actually contained one - something check #1 alone cannot show.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "audit_shuffle.csv"
DEFAULT_N_TRIALS = 200


@dataclass(frozen=True)
class AuditResult:
    real_r: float | None
    real_n: int
    trial_rs: list[float] = field(default_factory=list)
    n_trials_requested: int = 0
    n_trials_skipped: int = 0
    empirical_p: float | None = None

    @property
    def trial_mean(self) -> float | None:
        return sum(self.trial_rs) / len(self.trial_rs) if self.trial_rs else None

    @property
    def trial_min(self) -> float | None:
        return min(self.trial_rs) if self.trial_rs else None

    @property
    def trial_max(self) -> float | None:
        return max(self.trial_rs) if self.trial_rs else None


def shuffled_timestamp_map(headlines: list[Headline], seed: int) -> dict[tuple[str, str], datetime]:
    """Permute ``published_at`` across ``headlines``: the same multiset of
    timestamps, reassigned to different headlines, keyed by each headline's
    ``dedup_key()`` (source, link) so the map survives being looked up by a
    different ``Headline`` instance than the one that built it.

    A fixed ``seed`` makes a trial reproducible; different seeds give
    different permutations for the trial loop below.
    """
    rng = random.Random(seed)
    timestamps = [h.published_at for h in headlines]
    shuffled = timestamps[:]
    rng.shuffle(shuffled)
    return {h.dedup_key(): ts for h, ts in zip(headlines, shuffled)}


def correlation_for_rows(rows: list[dict]) -> float | None:
    """Pearson r of compound vs contemporaneous_return, or ``None`` if fewer
    than 2 rows resolved - not enough points for a correlation at all."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def run_shuffle_trial(in_path: Path, seed: int, live: bool = False) -> tuple[float | None, int]:
    """One shuffled-timestamp trial: re-run ``build_rows`` with every
    headline's ``published_at`` replaced by someone else's, via the
    ``timestamp_for`` hook. Returns ``(r, n)`` - ``r`` is ``None`` if fewer
    than 2 headlines still resolve to a priced session under the scrambled
    timestamp (a shuffled timestamp can land outside a ticker's one-month
    price fixture window, dropping that headline the same way a real
    ``PriceFetchError``/missing-bar would in ``build_rows``)."""
    headlines = read_csv(in_path)
    tmap = shuffled_timestamp_map(headlines, seed)
    rows, _ = build_rows(in_path, live=live, timestamp_for=lambda h: tmap[h.dedup_key()])
    return correlation_for_rows(rows), len(rows)


def audit(in_path: Path, n_trials: int, live: bool = False) -> AuditResult:
    real_rows, _ = build_rows(in_path, live=live)
    real_r = correlation_for_rows(real_rows)

    trial_rs: list[float] = []
    skipped = 0
    for seed in range(n_trials):
        r, n = run_shuffle_trial(in_path, seed, live=live)
        if r is None:
            skipped += 1
            continue
        trial_rs.append(r)

    if real_r is not None and trial_rs:
        at_least_as_extreme = sum(1 for r in trial_rs if abs(r) >= abs(real_r))
        empirical_p = at_least_as_extreme / len(trial_rs)
    else:
        empirical_p = None

    return AuditResult(
        real_r=real_r,
        real_n=len(real_rows),
        trial_rs=trial_rs,
        n_trials_requested=n_trials,
        n_trials_skipped=skipped,
        empirical_p=empirical_p,
    )


def write_trials_csv(result: AuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "shuffled_r"])
        for seed, r in enumerate(result.trial_rs):
            writer.writerow([seed, r])


def run(in_path: Path, out_path: Path, live: bool, n_trials: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = audit(in_path, n_trials=n_trials, live=live)
    write_trials_csv(result, out_path)

    if result.real_r is None:
        print(f"real pipeline: fewer than 2 resolved headlines (n={result.real_n}) - no correlation to audit")
        return 1

    print(f"real pipeline:   r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled control: {len(result.trial_rs)}/{n_trials} trials usable "
        f"({result.n_trials_skipped} skipped - shuffled timestamp fell outside a ticker's price fixture window)"
    )
    if result.trial_rs:
        print(
            f"                  mean r={result.trial_mean:+.3f}  "
            f"range [{result.trial_min:+.3f}, {result.trial_max:+.3f}]"
        )
        print(
            f"                  {sum(1 for r in result.trial_rs if abs(r) >= abs(result.real_r))}/"
            f"{len(result.trial_rs)} shuffled trials are at least as extreme as the real |r| "
            f"(empirical p={result.empirical_p:.3f})"
        )
        if result.empirical_p is not None and result.empirical_p < 0.05:
            print(
                "WARNING: the real correlation looks unusual against its own shuffled null - "
                "investigate before trusting it; this is what a real leak would look like.",
                file=sys.stderr,
            )
        else:
            print(
                "real r sits inside the shuffled null distribution - consistent with Day 5's finding "
                "that there is no detectable signal here to leak in the first place."
            )
    print(f"trial-by-trial r values written to {out_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="per-trial r values CSV to write")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="number of independent timestamp-shuffle trials to run",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.live, args.n_trials))


if __name__ == "__main__":
    main()
