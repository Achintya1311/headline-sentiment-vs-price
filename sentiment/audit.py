"""Day 8 CLI: ml-pipeline-audit - shuffle headline timestamps and confirm the
sentiment/return signal disappears.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --trials 2000 --seed 1

This is the "Correctness gate" README's Ground rules call out: a result that
has not passed it is a draft. The idea: a headline's VADER ``compound`` score
is fixed text content, but which trading session it gets paired with (Day
4's ``align_headline``) depends entirely on ``published_at``. If some bug let
information flow the wrong way - the aligned session including price action
that predates the headline, or some other channel pairing content with a
return it should not see - then even a *wrong* timestamp ought to produce
about the same correlation, because the leak does not care which timestamp
produced which session. Reshuffling ``published_at`` among the real
headlines (same multiset of timestamps, randomly reassigned) breaks the one
honest channel - "this headline happened to land before/after this
session's open" - while leaving everything else (text, scores, which company
each headline names) untouched. If the real correlation is not distinguishable
from the distribution of correlations a random relabelling produces, the
pipeline is not finding its signal through a back door.

Every trial reruns ``sentiment.correlate.build_rows_from_headlines`` - the
exact function Day 5's CLI uses - not a reimplementation, so this tests the
real pipeline rather than a model of it.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "audit_shuffle.csv"
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 0
MIN_N_FOR_CORRELATION = 2

ROW_FIELDNAMES = ["trial", "n_contemporaneous", "contemporaneous_r", "n_lagged", "lagged_r"]


@dataclass(frozen=True)
class TrialResult:
    n_contemporaneous: int
    contemporaneous_r: float | None
    n_lagged: int
    lagged_r: float | None


@dataclass(frozen=True)
class AuditReport:
    real: TrialResult
    trials: list[TrialResult]
    p_value_contemporaneous: float | None
    p_value_lagged: float | None


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` values permuted
    among them. Same multiset of timestamps (so the pre-open/intraday/
    post-close mix - and therefore the session dates in play - is unchanged
    in aggregate), but which headline gets which timestamp is randomised, so
    any correlation that depended on a *specific* headline landing at a
    *specific* time should wash out."""
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_timestamps)]


def _trial_from_rows(rows: list[dict]) -> TrialResult:
    compounds = [r["compound"] for r in rows]
    contemporaneous = [r["contemporaneous_return"] for r in rows]
    lagged_rows = [r for r in rows if r["lagged_return"] is not None]

    contemp_r = pearson_r(compounds, contemporaneous) if len(rows) >= MIN_N_FOR_CORRELATION else None
    lagged_r = (
        pearson_r([r["compound"] for r in lagged_rows], [r["lagged_return"] for r in lagged_rows])
        if len(lagged_rows) >= MIN_N_FOR_CORRELATION
        else None
    )
    return TrialResult(
        n_contemporaneous=len(rows),
        contemporaneous_r=contemp_r,
        n_lagged=len(lagged_rows),
        lagged_r=lagged_r,
    )


def permutation_p_value(real_stat: float, trial_stats: list[float]) -> float:
    """Two-sided empirical p-value: the fraction of shuffled trials whose
    statistic is at least as extreme (by absolute value) as the real one.

    High means the real result is unremarkable next to pure noise - what a
    pipeline with no hidden leak should show when its real signal is this
    weak. Low would mean the real result stands out from randomly-relabelled
    controls - evidence of a genuine timing-dependent effect, or of leakage,
    which this audit cannot by itself tell apart from each other; it can only
    tell a real result apart from noise.
    """
    if not trial_stats:
        raise ValueError("need at least one trial statistic")
    hits = sum(1 for t in trial_stats if abs(t) >= abs(real_stat))
    return hits / len(trial_stats)


def run_audit(
    in_path: Path, live: bool, trials: int, seed: int
) -> tuple[AuditReport, list[tuple[str, str]]]:
    headlines = read_csv(in_path)
    real_rows, unresolved = build_rows_from_headlines(headlines, live=live)
    real = _trial_from_rows(real_rows)

    rng = random.Random(seed)
    trial_results: list[TrialResult] = []
    for _ in range(trials):
        shuffled = shuffle_published_at(headlines, rng)
        shuffled_rows, _ = build_rows_from_headlines(shuffled, live=live)
        trial_results.append(_trial_from_rows(shuffled_rows))

    contemp_trial_rs = [t.contemporaneous_r for t in trial_results if t.contemporaneous_r is not None]
    lagged_trial_rs = [t.lagged_r for t in trial_results if t.lagged_r is not None]

    p_contemp = (
        permutation_p_value(real.contemporaneous_r, contemp_trial_rs)
        if real.contemporaneous_r is not None and contemp_trial_rs
        else None
    )
    p_lagged = (
        permutation_p_value(real.lagged_r, lagged_trial_rs)
        if real.lagged_r is not None and lagged_trial_rs
        else None
    )

    return (
        AuditReport(
            real=real,
            trials=trial_results,
            p_value_contemporaneous=p_contemp,
            p_value_lagged=p_lagged,
        ),
        unresolved,
    )


def write_trials_csv(report: AuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "trial": "real",
                "n_contemporaneous": report.real.n_contemporaneous,
                "contemporaneous_r": report.real.contemporaneous_r,
                "n_lagged": report.real.n_lagged,
                "lagged_r": report.real.lagged_r,
            }
        )
        for i, t in enumerate(report.trials):
            writer.writerow(
                {
                    "trial": i,
                    "n_contemporaneous": t.n_contemporaneous,
                    "contemporaneous_r": t.contemporaneous_r,
                    "n_lagged": t.n_lagged,
                    "lagged_r": t.lagged_r,
                }
            )


def _fmt_r(r: float | None) -> str:
    return f"{r:+.3f}" if r is not None else "n/a"


def _fmt_p(p: float | None) -> str:
    return f"{p:.3f}" if p is not None else "n/a"


def run(in_path: Path, out_path: Path, live: bool, trials: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    report, unresolved = run_audit(in_path, live, trials, seed)
    write_trials_csv(report, out_path)

    print(f"real pipeline: n={report.real.n_contemporaneous} resolved headlines ({out_path})")
    if unresolved:
        print(f"{len(unresolved)} headline(s) named a company with no resolvable ticker (same as Day 5)")
    print(
        f"real contemporaneous r={_fmt_r(report.real.contemporaneous_r)}  "
        f"real lagged r={_fmt_r(report.real.lagged_r)}"
    )
    print(f"{trials} shuffled-timestamp trials (seed={seed})")
    print(
        f"permutation p-value (contemporaneous): {_fmt_p(report.p_value_contemporaneous)}  "
        f"(fraction of shuffled trials at least as extreme as the real r)"
    )
    print(f"permutation p-value (lagged):          {_fmt_p(report.p_value_lagged)}")

    if report.p_value_contemporaneous is not None and report.p_value_contemporaneous < 0.05:
        print(
            "WARNING: the real contemporaneous correlation is more extreme than 95% of "
            "shuffled-timestamp controls - investigate before trusting it; this is what a "
            "leak would look like."
        )
    else:
        print(
            "the real result is not distinguishable from a random relabelling of the same "
            "headlines - exactly what 'no detectable signal yet' should look like, and the "
            "leakage test this repo's Correctness gate requires."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="per-trial CSV to write")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of shuffled-timestamp trials")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible audit")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.live, args.trials, args.seed))


if __name__ == "__main__":
    main()
