"""Day 8 CLI: the ml-pipeline-audit leakage test.

    python -m sentiment.audit
    python -m sentiment.audit --n-perm 2000 --seed 1

NEXT_STEPS.md's "Done when" is specific: *shuffle the headline timestamps
and the signal must disappear. If a shuffled-timestamp control still
predicts returns, the pipeline is leaking and the result is an artifact.*

What "shuffle the timestamps" means here: take the committed headlines and
randomly permute ``published_at`` **across headlines**, leaving title,
source, link and (therefore) the company/ticker each headline resolves to
untouched. Day 4's ``align_headline`` then assigns each headline to a
different, effectively random, trading session - so whatever return gets
paired with a headline's sentiment score no longer has anything to do with
when that headline was actually published. Re-run Day 5's two statistics
(contemporaneous Pearson r, and the high-vs-low-magnitude event-study mean
difference) on many independent shuffles. If this pipeline's apparent
"signal" really depends on correct timing, it should vanish under the
shuffle; if a shuffle still produces a "significant" result (its own 95%
CI, computed the same way Day 5 computes it, excludes zero) about as often
as the real, correctly-timed data does, that is not noise - it means
something other than news timing is driving the number (most plausibly: a
handful of tickers with a persistent trend over the fixture window,
independent of any specific headline date - see the README's Day 8
Findings).

This is a calibration check on Day 5's own significance test, not a
one-off comparison: under a true null (no leakage), roughly 5% of 95%-CI
shuffles should show "significant" results by chance alone. A much higher
rate across hundreds of shuffles is the leak signature this script exists
to catch, which is why it runs as an assertion in the test suite
(``tests/test_audit.py``), not just as something read off the console
once.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_EVENT_THRESHOLD, DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import bootstrap_mean_diff_ci, pearson_with_ci

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "pipeline_audit.csv"
DEFAULT_N_PERM = 500
DEFAULT_SEED = 0


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` values randomly
    permuted *across* headlines (a permutation, not independent resampling,
    so the real multiset of publish times is preserved - only which headline
    got which timestamp changes). Title, source, link and ``scraped_at``
    (and therefore ticker resolution) are left exactly as scraped."""
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


def contemporaneous_r(rows: list[dict]) -> tuple[float, float, float, int] | None:
    """(r, ci_low, ci_high, n) for compound vs contemporaneous_return, or
    ``None`` if fewer than 2 resolved rows."""
    if len(rows) < 2:
        return None
    stat = pearson_with_ci([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])
    return stat.r, stat.ci_low, stat.ci_high, stat.n


def event_study_diff(rows: list[dict], event_threshold: float) -> tuple[float, float, float, int, int] | None:
    """(diff, ci_low, ci_high, n_high, n_low) for the high-vs-low-magnitude
    mean contemporaneous return, or ``None`` if either group is empty."""
    high = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) > event_threshold]
    low = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) <= event_threshold]
    if not high or not low:
        return None
    result = bootstrap_mean_diff_ci(high, low)
    return result.diff, result.ci_low, result.ci_high, result.n_a, result.n_b


@dataclass(frozen=True)
class PermutationRun:
    """One shuffle's result for both audited statistics. Either field is
    ``None`` if that shuffle left too few rows (a ticker's bars ran out at
    the newly-assigned session date) to compute it."""

    contemporaneous: tuple[float, float, float, int] | None
    event: tuple[float, float, float, int, int] | None


@dataclass(frozen=True)
class AuditResult:
    real_contemporaneous: tuple[float, float, float, int] | None
    real_event: tuple[float, float, float, int, int] | None
    shuffles: list[PermutationRun]

    def _significant_rate(self, extractor) -> tuple[int, int]:
        """(count significant, count computable) across all shuffles, where
        "significant" means the shuffle's own 95% CI excludes zero - the
        same test Day 5 applies to the real data."""
        significant = 0
        computable = 0
        for run in self.shuffles:
            stat = extractor(run)
            if stat is None:
                continue
            computable += 1
            _, ci_low, ci_high, *_ = stat
            if ci_low > 0 or ci_high < 0:
                significant += 1
        return significant, computable

    def contemporaneous_false_positive_rate(self) -> tuple[int, int]:
        return self._significant_rate(lambda run: run.contemporaneous)

    def event_false_positive_rate(self) -> tuple[int, int]:
        return self._significant_rate(lambda run: run.event)


def run_audit(
    headlines: list[Headline],
    event_threshold: float,
    n_perm: int,
    seed: int,
    live: bool = False,
) -> AuditResult:
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_contemporaneous = contemporaneous_r(real_rows)
    real_event = event_study_diff(real_rows, event_threshold)

    rng = random.Random(seed)
    shuffles: list[PermutationRun] = []
    for _ in range(n_perm):
        shuffled = shuffle_published_at(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live)
        shuffles.append(
            PermutationRun(
                contemporaneous=contemporaneous_r(rows),
                event=event_study_diff(rows, event_threshold),
            )
        )

    return AuditResult(real_contemporaneous=real_contemporaneous, real_event=real_event, shuffles=shuffles)


def write_shuffles_csv(result: AuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shuffle", "contemporaneous_r", "contemporaneous_significant", "event_diff", "event_significant"])
        for i, run in enumerate(result.shuffles):
            c = run.contemporaneous
            e = run.event
            writer.writerow(
                [
                    i,
                    c[0] if c else "",
                    (c[1] > 0 or c[2] < 0) if c else "",
                    e[0] if e else "",
                    (e[1] > 0 or e[2] < 0) if e else "",
                ]
            )


def run(in_path: Path, out_path: Path, live: bool, event_threshold: float, n_perm: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_audit(headlines, event_threshold, n_perm, seed, live=live)
    write_shuffles_csv(result, out_path)

    print(f"ml-pipeline-audit: {n_perm} shuffled-timestamp controls -> {out_path}")

    if result.real_contemporaneous:
        r, lo, hi, n = result.real_contemporaneous
        sig = "significant" if (lo > 0 or hi < 0) else "not significant"
        print(f"real data   contemporaneous: r={r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={n}  ({sig})")
    sig_count, computable = result.contemporaneous_false_positive_rate()
    if computable:
        rate = sig_count / computable
        print(f"shuffled    contemporaneous: {sig_count}/{computable} shuffles ({rate:.1%}) came back 'significant' by chance")

    if result.real_event:
        diff, lo, hi, n_high, n_low = result.real_event
        sig = "significant" if (lo > 0 or hi < 0) else "not significant"
        print(f"real data   event study:     diff={diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n_high={n_high} n_low={n_low}  ({sig})")
    sig_count, computable = result.event_false_positive_rate()
    if computable:
        rate = sig_count / computable
        print(f"shuffled    event study:     {sig_count}/{computable} shuffles ({rate:.1%}) came back 'significant' by chance")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="per-shuffle CSV to write")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--event-threshold", type=float, default=DEFAULT_EVENT_THRESHOLD, help="same meaning as sentiment.correlate")
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM, help="number of shuffled-timestamp controls to run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible audit")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.live, args.event_threshold, args.n_perm, args.seed))


if __name__ == "__main__":
    main()
