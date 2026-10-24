"""Day 8 follow-up: robustness checks the shuffled-timestamp audit cannot make.

    python -m sentiment.robustness
    python -m sentiment.robustness --n-perm 5000 --seed 1

``sentiment.audit`` and ``sentiment.leakage_audit`` only reshuffle time. They
cannot see these problems, so each gets its own check:

- **Reverse causality.** Some headlines name the price move ("X shares rise
  4%"). A correlation there may only show the headline reported the move.
  ``names_price_move`` flags them and the correlation is re-run without them.
- **Market-wide shocks.** Returns on the same session share that day's market
  move, so they are not independent draws. ``session_demeaned_r`` removes each
  session's mean return, and ``within_session_permutation`` shuffles returns
  only inside a session, so ticker identity is scrambled but the day effect is
  kept in the null.
- **Untrusted feed timestamps.** ``timestamp_integrity`` looks for headlines
  dated after they were scraped, duplicate titles or links with different
  times, and many headlines stamped to the same instant.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r, pearson_with_ci

# A move verb plus a percentage: "shares rise 4%", "rallies 89%", "fall 2%".
# "sees 41% upside" has no move verb, so a forecast is not flagged.
_MOVE = re.compile(
    r"\b(ris(?:e|es|ing)|fall(?:s|ing)?|rall(?:y|ies|ied)|jump(?:s|ed)?|surg(?:e|es|ed)|"
    r"slip(?:s|ped)?|slump(?:s|ed)?|drop(?:s|ped)?|tumbl(?:e|es|ed)|plung(?:e|es|ed)|"
    r"gain(?:s|ed)?|climb(?:s|ed)?|sink(?:s)?|sank)\b[^%]{0,40}\d+(?:\.\d+)?\s?%",
    re.IGNORECASE,
)
BATCH_STAMP_MIN = 5  # this many headlines on one exact timestamp looks batch-stamped


def names_price_move(title: str) -> bool:
    return bool(_MOVE.search(title))


def _r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    return pearson_r(xs, ys)


def session_demeaned_r(rows: list[dict], value_key: str = "contemporaneous_return") -> float:
    """Correlation of compound with ``value_key`` minus its session's mean.

    ``value_key`` defaults to ``contemporaneous_return`` (Day 8's original
    use). ``sentiment.prereg`` (Day 84, hub CRITIQUES.md R18) reuses this
    same demeaning on ``abnormal_return_0_1`` instead - the market-wide-day
    confound this function removes applies to any per-headline return
    column, not only the raw contemporaneous one.
    """
    by_day: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_day[r["session_date"]].append(r[value_key])
    means = {d: sum(v) / len(v) for d, v in by_day.items()}
    return _r([r["compound"] for r in rows], [r[value_key] - means[r["session_date"]] for r in rows])


@dataclass(frozen=True)
class WithinSessionResult:
    real_r: float
    null_mean: float
    null_std: float
    p_value: float
    n_perm: int


def within_session_permutation(rows: list[dict], n_perm: int = 2000, seed: int = 0) -> WithinSessionResult:
    """Shuffle returns among headlines of the same session. This scrambles which
    ticker's return each headline gets, keeps each day's market move in the
    null, and so asks whether the pairing beats chance given the day effect."""
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    real = _r(compounds, returns)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["session_date"]].append(i)
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        shuffled = list(returns)
        for idx in groups.values():
            vals = [returns[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                shuffled[i] = v
        null.append(_r(compounds, shuffled))
    mean = sum(null) / n_perm
    std = (sum((x - mean) ** 2 for x in null) / n_perm) ** 0.5
    # +1 smoothing: a finite permutation sample never proves p = 0
    p = (1 + sum(1 for x in null if abs(x) >= abs(real))) / (n_perm + 1)
    return WithinSessionResult(real, mean, std, p, n_perm)


def timestamp_integrity(headlines: list[Headline]) -> dict[str, list[str]]:
    """Return {check: examples}; an empty dict means every check passed."""
    issues: dict[str, list[str]] = {}
    future = [h.title for h in headlines if h.published_at > h.scraped_at]
    if future:
        issues["published after it was scraped"] = future
    seen_links: dict[str, set] = defaultdict(set)
    for h in headlines:
        seen_links[h.link].add(h.published_at)
    dup_links = [k for k, v in seen_links.items() if len(v) > 1]
    if dup_links:
        issues["same link, different published time"] = dup_links

    # A title repeat is only suspicious if its distinct published times
    # outnumber its distinct links: a recurring daily template ("X Share
    # Price Highlights") reused once per distinct article/link produces the
    # same title with several different, correct times and must not be
    # flagged. A genuine same-link retiming (fewer links than times) still is.
    by_title: dict[str, list[Headline]] = defaultdict(list)
    for h in headlines:
        by_title[h.title].append(h)
    dup_titles = []
    for title, hs in by_title.items():
        times = {h.published_at for h in hs}
        links = {h.link for h in hs}
        if len(times) > 1 and len(links) < len(times):
            dup_titles.append(title)
    if dup_titles:
        issues["same title, different published time"] = dup_titles
    stamps: dict = defaultdict(list)
    for h in headlines:
        stamps[h.published_at].append(h.title)
    batch = [f"{t.isoformat()} x{len(v)}" for t, v in stamps.items() if len(v) >= BATCH_STAMP_MIN]
    if batch:
        issues[f"{BATCH_STAMP_MIN}+ headlines on one exact timestamp"] = batch
    return issues


def run(in_path: Path, n_perm: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1
    rows, _, _ = build_rows(in_path)
    if len(rows) < 3:
        print("too few resolved headlines for these checks", file=sys.stderr)
        return 1
    print(f"all resolved headlines: n={len(rows)}  r={_r([r['compound'] for r in rows], [r['contemporaneous_return'] for r in rows]):+.3f}")

    kept = [r for r in rows if not names_price_move(r["title"])]
    print(f"\nreverse causality: {len(rows) - len(kept)} of {len(rows)} headlines name a price move")
    k = pearson_with_ci([r["compound"] for r in kept], [r["contemporaneous_return"] for r in kept])
    print(f"  without them: n={k.n}  r={k.r:+.3f}  95% CI [{k.ci_low:+.3f}, {k.ci_high:+.3f}]  distinct scores={len({r['compound'] for r in kept})}")

    print(f"\nmarket-wide shocks: {len({r['session_date'] for r in rows})} distinct sessions")
    print(f"  session-demeaned r = {session_demeaned_r(rows):+.3f}")
    w = within_session_permutation(rows, n_perm, seed)
    print(f"  within-session permutation ({w.n_perm}): real r={w.real_r:+.3f}  null mean={w.null_mean:+.3f} std={w.null_std:.3f}  p={w.p_value:.3f}")

    issues = timestamp_integrity(read_csv(in_path))
    print("\nfeed timestamps: " + ("all checks passed" if not issues else f"{len(issues)} check(s) flagged"))
    for check, examples in issues.items():
        print(f"  {check}: {len(examples)} (e.g. {examples[0][:70]})")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    raise SystemExit(run(a.in_path, a.n_perm, a.seed))


if __name__ == "__main__":
    main()
