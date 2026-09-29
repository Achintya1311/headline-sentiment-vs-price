"""Day 7 CLI: sentiment-price overlay and cumulative abnormal return charts.

    python -m sentiment.overlay
    python -m sentiment.overlay --live
    python -m sentiment.overlay --event-threshold 0.3 --window-before 5 --window-after 1

Two charts, both written to ``outputs/`` (never committed - see
``.gitignore`` - and never a dashboard, per the portfolio's CLI-only rule):

1. **Sentiment-price overlay** (``outputs/sentiment_price_overlay.png``): one
   panel per "high-magnitude" headline (Day 5's ``|compound| > event
   threshold`` group - the only 3 headlines in this fixture that carry
   sentiment content beyond the saturated "Share Price Highlights" template,
   see ``sentiment.correlate``'s Findings). Each panel plots that ticker's
   full ~1-month cumulative return with a marker on the headline's aligned
   session date. The other 20 resolved headlines are deliberately left out
   of this chart: they all share the identical VADER compound (0.296), so a
   panel per headline would be visually indistinguishable noise - see the
   README for why, and the CAR chart below for where they are used instead.

2. **Cumulative abnormal return** (``outputs/cumulative_abnormal_return.png``):
   for every resolved headline, alignes its ticker's daily returns to
   headline-relative trading-day offsets (0 = the headline's aligned
   session), computes each day's abnormal return against
   ``sentiment.market``'s equal-weighted proxy, cumulates it across the
   window, and averages across headlines *within* the high- and
   low-magnitude groups separately - the same two groups Day 5's event study
   already used, now shown as a trajectory instead of a single before/after
   number.

Neither chart is the leakage test (that is CI's job, see README) - both are
descriptive views of the same rows ``sentiment.correlate.build_rows``
already produces.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CLI-only: never opens a window, always writes a file
import matplotlib.pyplot as plt

from sentiment.correlate import DEFAULT_EVENT_THRESHOLD, DEFAULT_IN, build_rows
from sentiment.market import equal_weighted_daily_returns, load_universe_bars
from sentiment.prices import Bar, load_bars

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OVERLAY_OUT = OUT_DIR / "sentiment_price_overlay.png"
CAR_OUT = OUT_DIR / "cumulative_abnormal_return.png"

DEFAULT_WINDOW_BEFORE = 5
DEFAULT_WINDOW_AFTER = 1


def cumulative_return_series(bars: list[Bar]) -> tuple[list[date], list[float]]:
    """Cumulative return rebased to 0.0 at the first bar (compounded, not
    additive - this is a full ~1-month series, long enough that compounding
    and summing daily returns would visibly diverge)."""
    dates: list[date] = []
    cum: list[float] = []
    acc = 1.0
    for b in bars:
        acc *= 1 + b.session_return
        dates.append(b.date)
        cum.append(acc - 1.0)
    return dates, cum


def event_window_abnormal_returns(
    bars: list[Bar],
    event_date: date,
    market_returns: dict[date, float],
    window_before: int,
    window_after: int,
) -> list[tuple[int, float]]:
    """Abnormal return (ticker return - market proxy return) at each trading
    -day offset from ``event_date`` (0 = the event session itself), for
    whichever offsets in [-window_before, +window_after] this ticker's bars
    and the market proxy both have data for. Offsets are trading-day counts
    within ``bars``, not calendar days, so weekends/holidays never appear as
    a false multi-day gap.

    Returns an empty list if ``event_date`` is not one of ``bars`` own dates
    (should not happen for a row ``sentiment.correlate`` already aligned and
    fetched a bar for, but this stays defensive rather than assuming it)."""
    index_by_date = {b.date: i for i, b in enumerate(bars)}
    if event_date not in index_by_date:
        return []
    event_idx = index_by_date[event_date]

    result: list[tuple[int, float]] = []
    for offset in range(-window_before, window_after + 1):
        idx = event_idx + offset
        if idx < 0 or idx >= len(bars):
            continue
        bar = bars[idx]
        market_return = market_returns.get(bar.date)
        if market_return is None:
            continue
        result.append((offset, bar.session_return - market_return))
    return result


def cumulative_by_offset(offset_returns: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Running sum of abnormal returns in ascending offset order - the
    cumulative abnormal return (CAR) at each offset. Additive, the standard
    approximation for a short event window, not compounded like the
    full-series overlay chart above."""
    ordered = sorted(offset_returns, key=lambda pair: pair[0])
    car: list[tuple[int, float]] = []
    running = 0.0
    for offset, abn in ordered:
        running += abn
        car.append((offset, running))
    return car


def split_by_magnitude(rows: list[dict], event_threshold: float) -> tuple[list[dict], list[dict]]:
    high = [r for r in rows if abs(r["compound"]) > event_threshold]
    low = [r for r in rows if abs(r["compound"]) <= event_threshold]
    return high, low


def average_car_by_offset(
    rows: list[dict],
    bars_by_ticker: dict[str, list[Bar]],
    market_returns: dict[date, float],
    window_before: int,
    window_after: int,
) -> dict[int, tuple[float, int]]:
    """Mean CAR and headline count at each offset, across ``rows``. Rows
    whose ticker is missing a bar at a given offset (most commonly: the
    headline's aligned session is the last date in the price fixture, so
    offset +1 has no next session yet - the same gap Day 5's and Day 6's
    ``lagged_return`` already documented) simply do not contribute at that
    offset, so ``n`` can shrink near the edges of the window rather than
    staying constant across it."""
    by_offset: dict[int, list[float]] = {}
    for row in rows:
        bars = bars_by_ticker.get(row["ticker"])
        if not bars:
            continue
        event_date = date.fromisoformat(row["session_date"])
        offset_returns = event_window_abnormal_returns(bars, event_date, market_returns, window_before, window_after)
        for offset, car in cumulative_by_offset(offset_returns):
            by_offset.setdefault(offset, []).append(car)
    return {offset: (sum(vals) / len(vals), len(vals)) for offset, vals in sorted(by_offset.items())}


def plot_overlay(rows: list[dict], bars_by_ticker: dict[str, list[Bar]], out_path: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, row in zip(axes[0], rows):
        ticker = row["ticker"]
        bars = bars_by_ticker[ticker]
        dates, cum = cumulative_return_series(bars)
        ax.plot(dates, [c * 100 for c in cum], color="#1f77b4", linewidth=1.5)

        event_date = date.fromisoformat(row["session_date"])
        event_idx = next((i for i, d in enumerate(dates) if d == event_date), None)
        if event_idx is not None:
            color = "#2ca02c" if row["compound"] > 0 else "#d62728"
            ax.axvline(event_date, color=color, linestyle="--", linewidth=1, alpha=0.7)
            ax.scatter([event_date], [cum[event_idx] * 100], color=color, zorder=5, s=60)
            ax.annotate(
                f"compound={row['compound']:+.2f}",
                (event_date, cum[event_idx] * 100),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=8,
            )

        ax.set_title(f"{row['company']} ({ticker})", fontsize=10)
        ax.set_ylabel("cumulative return (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Sentiment-price overlay: high-magnitude headlines (|compound| > threshold)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_car(
    high_car: dict[int, tuple[float, int]],
    low_car: dict[int, tuple[float, int]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, series, color in (("high-magnitude", high_car, "#d62728"), ("low-magnitude", low_car, "#1f77b4")):
        if not series:
            continue
        offsets = sorted(series)
        ax.plot(offsets, [series[o][0] * 100 for o in offsets], marker="o", color=color, label=label)

    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("trading days relative to headline (0 = aligned session)")
    ax.set_ylabel("mean cumulative abnormal return (%)")
    ax.set_title("Cumulative abnormal return: high- vs low-magnitude sentiment headlines")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(
    in_path: Path,
    out_dir: Path,
    live: bool,
    event_threshold: float,
    window_before: int,
    window_after: int,
) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    rows, _ = build_rows(in_path, live=live)
    if not rows:
        print("no headlines resolved to a fetchable ticker; nothing to chart", file=sys.stderr)
        return 1

    tickers = sorted({r["ticker"] for r in rows})
    bars_by_ticker = {t: load_bars(t, live=live) for t in tickers}

    universe = load_universe_bars()
    market_returns = equal_weighted_daily_returns(universe)

    high, low = split_by_magnitude(rows, event_threshold)

    overlay_path = out_dir / "sentiment_price_overlay.png"
    car_path = out_dir / "cumulative_abnormal_return.png"

    if high:
        plot_overlay(high, bars_by_ticker, overlay_path)
        print(f"sentiment-price overlay ({len(high)} high-magnitude headlines) -> {overlay_path}")
    else:
        print(f"no headlines exceed |compound| > {event_threshold}; overlay chart skipped", file=sys.stderr)

    high_car = average_car_by_offset(high, bars_by_ticker, market_returns, window_before, window_after)
    low_car = average_car_by_offset(low, bars_by_ticker, market_returns, window_before, window_after)
    if high_car or low_car:
        plot_car(high_car, low_car, car_path)
        print(f"cumulative abnormal return (high n={len(high)}, low n={len(low)}) -> {car_path}")
        for label, series in (("high", high_car), ("low", low_car)):
            if 0 in series:
                mean, n = series[0]
                print(f"  {label}-magnitude CAR at offset 0: {mean:+.4f} (n={n})")
    else:
        print("no offsets had both ticker and market-proxy data; CAR chart skipped", file=sys.stderr)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="directory to write charts into")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD,
        help="|compound| above this is 'high-magnitude', same default as sentiment.correlate",
    )
    parser.add_argument("--window-before", type=int, default=DEFAULT_WINDOW_BEFORE, help="trading days before the event to include")
    parser.add_argument("--window-after", type=int, default=DEFAULT_WINDOW_AFTER, help="trading days after the event to include")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_dir, args.live, args.event_threshold, args.window_before, args.window_after))


if __name__ == "__main__":
    main()
