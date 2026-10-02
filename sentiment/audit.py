"""Day 8 CLI: shuffled-timestamp leakage control, the project's correctness gate.

    python -m sentiment.audit

Every earlier day's "no signal" result in this repo (Day 5's near-zero
correlation, Day 6's zero-variance regression, Day 7's descriptive-only
charts) could mean two very different things: (a) sentiment genuinely has
no predictive power on this fixture, or (b) the alignment/correlation
pipeline has a look-ahead bug that just happens not to manufacture *extra*
signal on this particular data. A null result on real data cannot tell
those two apart by itself - there is nothing there to lose if real data
never had a signal to begin with. See "Why this might be spurious" in the
README.

So this audit runs two checks, and only the first is a gate CI can fail on:

1. **Planted-signal control** (``run_planted_signal_control``). Build a
   synthetic set of headlines on distinct trading days, each published
   before its own session's open. Define a "planted" session return that is
   a deterministic function of that same headline's sentiment score
   (``PLANTED_BETA * compound``, plus a small jitter) - a real relationship,
   by construction, routed through the real ``sentiment.market_hours.
   align_headline``, not a stand-in. Correlating sentiment against the
   planted return should find it: a strong, clearly non-zero ``r``. Then
   ``shuffle_timestamps`` reassigns every headline's ``published_at`` to a
   *different* headline's timestamp (a random derangement - nothing keeps
   its own timestamp) and the same rows are rebuilt. Because the planted
   return is keyed to the session each headline's *own* timestamp resolves
   to, swapping timestamps swaps which planted return each sentiment score
   gets paired with - correlation should collapse towards zero. If it
   doesn't, the pipeline is carrying the compound/return pairing through
   some channel other than the aligned session date, which is exactly the
   look-ahead failure mode this test exists to catch. This check fails the
   build (exit 1) if the signal isn't detected pre-shuffle or doesn't
   collapse post-shuffle.

2. **Real-fixture control** (``run_real_fixture_control``). Runs the
   identical timestamp shuffle through the real pipeline
   (``sentiment.correlate.build_rows_from_headlines``) against the real
   scraped headlines and the real committed price fixtures. This is
   informational only, never a gate: Day 5-7 already found ~no correlation
   on real data, so a null result before *and* after shuffling proves
   nothing new about leakage by itself - there was nothing to lose. It is
   reported for completeness, not treated as evidence either way.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import read_csv
from sentiment.market_hours import IST, align_headline, next_trading_day
from sentiment.stats import PearsonResult, pearson_with_ci

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

N_SYNTHETIC = 60
PLANTED_BETA = 0.05  # planted return per unit of compound
NOISE_SD = 0.002  # small jitter so the fit isn't a mathematically exact line
SYNTHETIC_START_DAY = date(2026, 1, 5)  # a Monday, arbitrary but fixed

# Thresholds for the planted-signal gate. The construction above produces an
# unshuffled |r| close to 1.0 and a shuffled |r| close to 0.0 - these leave a
# wide margin either side rather than asserting the exact numbers, since the
# precise value depends on the noise draw.
DETECTED_R = 0.6
COLLAPSED_R = 0.4


def shuffle_timestamps(timestamps: list[datetime], seed: int = 0) -> list[datetime]:
    """Reassign every timestamp to a *different* position via a random
    derangement (Sattolo's algorithm: a uniformly random single n-cycle).
    Plain ``random.shuffle`` can leave some elements in place by chance;
    this control needs every headline to lose its own timestamp, not most
    of them."""
    rng = random.Random(seed)
    order = list(range(len(timestamps)))
    for i in range(len(order) - 1, 0, -1):
        j = rng.randrange(0, i)
        order[i], order[j] = order[j], order[i]
    return [timestamps[order[i]] for i in range(len(order))]


def synthetic_headlines(n: int = N_SYNTHETIC, seed: int = 0) -> tuple[list[datetime], list[float]]:
    """``n`` distinct trading days, one synthetic headline each, timestamped
    before that day's own open (pre-open, per ``market_hours`` - so each
    aligns to the same calendar day it was published on, deterministically).
    Compound scores are spread uniformly over [-1, 1]; nothing here is read
    from a title, so no VADER/FinBERT scoring is involved."""
    rng = random.Random(seed)
    timestamps = []
    day = SYNTHETIC_START_DAY
    for _ in range(n):
        timestamps.append(datetime.combine(day, time(9, 0), tzinfo=IST))
        day = next_trading_day(day)
    compounds = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    return timestamps, compounds


def planted_return_table(
    timestamps: list[datetime], compounds: list[float], seed: int = 1
) -> dict[date, float]:
    """One planted return per session, keyed by the session each headline's
    own timestamp resolves to: ``PLANTED_BETA * compound + noise``. A real,
    if synthetic, relationship between sentiment and return - the thing the
    shuffle control below is supposed to be able to destroy."""
    rng = random.Random(seed)
    table: dict[date, float] = {}
    for ts, c in zip(timestamps, compounds):
        session = align_headline(ts).session_date
        table[session] = PLANTED_BETA * c + rng.gauss(0, NOISE_SD)
    return table


def planted_rows(
    timestamps: list[datetime], compounds: list[float], table: dict[date, float]
) -> list[tuple[float, float]]:
    """(compound, return) pairs after aligning each timestamp to its session
    and looking up that session's planted return. A timestamp that resolves
    to a session with no entry in ``table`` is dropped (can only happen if
    ``timestamps``/``table`` come from different synthetic runs)."""
    rows = []
    for ts, c in zip(timestamps, compounds):
        session = align_headline(ts).session_date
        ret = table.get(session)
        if ret is not None:
            rows.append((c, ret))
    return rows


@dataclass(frozen=True)
class ControlResult:
    label: str
    unshuffled: PearsonResult | None
    shuffled: PearsonResult | None


def _correlate_or_none(rows: list[tuple[float, float]]) -> PearsonResult | None:
    if len(rows) < 4:
        return None
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    return pearson_with_ci(xs, ys)


def run_planted_signal_control(seed: int = 0, shuffle_seed: int = 99) -> ControlResult:
    timestamps, compounds = synthetic_headlines(seed=seed)
    table = planted_return_table(timestamps, compounds, seed=seed + 1)

    unshuffled_rows = planted_rows(timestamps, compounds, table)
    shuffled_timestamps = shuffle_timestamps(timestamps, seed=shuffle_seed)
    shuffled_rows = planted_rows(shuffled_timestamps, compounds, table)

    return ControlResult(
        label="planted-signal control (synthetic headlines, synthetic returns)",
        unshuffled=_correlate_or_none(unshuffled_rows),
        shuffled=_correlate_or_none(shuffled_rows),
    )


def run_real_fixture_control(in_path: Path = DEFAULT_IN, live: bool = False, seed: int = 7) -> ControlResult:
    headlines = read_csv(in_path)
    real_rows, _ = build_rows_from_headlines(headlines, live=live)

    shuffled_timestamps = shuffle_timestamps([h.published_at for h in headlines], seed=seed)
    shuffled_headlines = [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_timestamps)]
    shuffled_rows, _ = build_rows_from_headlines(shuffled_headlines, live=live)

    def as_pairs(rows: list[dict]) -> list[tuple[float, float]]:
        return [(r["compound"], r["contemporaneous_return"]) for r in rows]

    return ControlResult(
        label="real-fixture control (real headlines, real price fixtures)",
        unshuffled=_correlate_or_none(as_pairs(real_rows)),
        shuffled=_correlate_or_none(as_pairs(shuffled_rows)),
    )


def planted_signal_passed(result: ControlResult) -> bool:
    if result.unshuffled is None or result.shuffled is None:
        return False
    return abs(result.unshuffled.r) >= DETECTED_R and abs(result.shuffled.r) <= COLLAPSED_R


def _fmt(stat: PearsonResult | None) -> str:
    if stat is None:
        return "not enough rows for a correlation"
    return f"r={stat.r:+.3f}  95% CI [{stat.ci_low:+.3f}, {stat.ci_high:+.3f}]  n={stat.n}"


def print_report(planted: ControlResult, real: ControlResult, passed: bool) -> None:
    print(f"{planted.label}")
    print(f"  unshuffled: {_fmt(planted.unshuffled)}")
    print(f"  shuffled:   {_fmt(planted.shuffled)}")
    print(f"  gate: {'PASS' if passed else 'FAIL'} (detect |r|>={DETECTED_R}, collapse |r|<={COLLAPSED_R})")
    print()
    print(f"{real.label}  [informational only - not a gate, see module docstring]")
    print(f"  unshuffled: {_fmt(real.unshuffled)}")
    print(f"  shuffled:   {_fmt(real.shuffled)}")


def write_markdown(planted: ControlResult, real: ControlResult, passed: bool, path: Path, generated: str) -> None:
    lines = [
        f"# Pipeline audit — {generated}",
        "",
        "Shuffled-timestamp leakage control (Day 8). See `sentiment/audit.py` "
        "for what each check does and why only the first is a gate.",
        "",
        f"## {planted.label}",
        f"- unshuffled: {_fmt(planted.unshuffled)}",
        f"- shuffled: {_fmt(planted.shuffled)}",
        f"- gate: **{'PASS' if passed else 'FAIL'}**",
        "",
        f"## {real.label} (informational only)",
        f"- unshuffled: {_fmt(real.unshuffled)}",
        f"- shuffled: {_fmt(real.shuffled)}",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="real headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch real prices instead of reading fixtures")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--date", default=None, help="override the generated date (mainly for tests)")
    args = parser.parse_args(argv)

    generated = args.date or date.today().isoformat()

    planted = run_planted_signal_control()
    real = run_real_fixture_control(in_path=args.in_path, live=args.live)
    passed = planted_signal_passed(planted)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(planted, real, passed, out_dir / f"audit_{generated}.md", generated)

    print_report(planted, real, passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
