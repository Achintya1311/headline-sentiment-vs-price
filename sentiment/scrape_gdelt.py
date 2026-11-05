"""Day 84 CLI (hub CRITIQUES.md R18): scrape headlines from the GDELT
Project's free DOC API, one query per company this project already knows
how to resolve to a priced ticker (``sentiment.tickers.known_companies``).

Offline by default - reads whatever committed snapshot exists under
``fixtures/gdelt/<company_slug>.json`` for each company, same convention as
``sentiment.scrape``'s offline path. A company with no committed snapshot
yet is skipped, not a failure - it is a future ``--live`` candidate. The
one committed fixture today (``fixtures/gdelt/infosys.json``) is a trimmed
real response, captured 24 Oct 2026, used for tests and this day's own
verification.

    python -m sentiment.scrape_gdelt            # offline: writes outputs/headlines_gdelt_offline.csv
    python -m sentiment.scrape_gdelt --live     # appends to fixtures/headlines/headlines_raw.csv
    python -m sentiment.scrape_gdelt --company Infosys --live
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sentiment.gdelt import fetch_gdelt_live_raw, parse_gdelt_response
from sentiment.headline import Headline, merge_and_write, read_csv
from sentiment.robots import RateLimiter, RobotsCache
from sentiment.tickers import known_companies

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "gdelt"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
# P110: the offline path replays committed test snapshots; merging them into
# the real corpus would add a source by hand, so it writes here by default.
OFFLINE_OUT = Path(__file__).resolve().parent.parent / "outputs" / "headlines_gdelt_offline.csv"


def _slug(company: str) -> str:
    return company.lower().replace(" ", "_")


def fixture_path(company: str) -> Path:
    return FIXTURES_DIR / f"{_slug(company)}.json"


def scrape_company_offline(company: str) -> list[Headline]:
    path = fixture_path(company)
    if not path.exists():
        raise FileNotFoundError(
            f"no committed GDELT fixture for {company!r} at {path}; run with --live to fetch one"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    scraped_at = datetime.now(timezone.utc)
    return parse_gdelt_response(data, scraped_at=scraped_at)


def run(companies: list[str], live: bool, out: Path, snapshot: bool = True) -> int:
    robots = RobotsCache()
    limiter = RateLimiter()
    all_headlines: list[Headline] = []
    exit_code = 0

    skipped: dict[str, int] = {}
    if not live and out.resolve() == DEFAULT_OUT.resolve():
        print(
            f"refusing to merge offline GDELT test snapshots into the committed corpus {out} (P110); "
            f"pass another --out, or --live",
            file=sys.stderr,
        )
        return 2
    for company in companies:
        if live:
            headlines, status, raw = fetch_gdelt_live_raw(company, robots, limiter)
            if status != "ok":
                print(f"[{company}] SKIPPED ({status})", file=sys.stderr)
                kind = status.split(":", 1)[0]
                skipped[kind] = skipped.get(kind, 0) + 1
                if status == "robots_denied":
                    continue
                exit_code = 1
                continue
            if snapshot and raw is not None:
                # Write the bytes already fetched (P160): a second request
                # for the snapshot hit GDELT's rate limit and its uncaught
                # error lost every headline parsed so far. A failed write
                # costs the snapshot only, never the headlines.
                path = fixture_path(company)
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                except OSError as e:
                    print(f"[{company}] snapshot not written ({e})", file=sys.stderr)
        else:
            try:
                headlines = scrape_company_offline(company)
                status = "ok (fixture)"
            except FileNotFoundError as e:
                print(f"[{company}] SKIPPED (no fixture: {e})", file=sys.stderr)
                continue
        print(f"[{company}] {status}: {len(headlines)} headlines")
        all_headlines.extend(headlines)

    if live and skipped:
        # R27: one line a workflow log reader cannot miss, so a source that
        # silently returns nothing every day is visible.
        fetched = len(companies) - sum(skipped.values())
        kinds = ", ".join(f"{n} {k}" for k, n in sorted(skipped.items()))
        print(f"GDELT: {fetched} of {len(companies)} queries fetched ({kinds})", file=sys.stderr)

    prior_count = len(read_csv(out)) if out.exists() else 0
    merged = merge_and_write(out, all_headlines)
    new_count = len(merged) - prior_count
    print(f"scraped {len(all_headlines)}, {new_count} new after dedup, {len(merged)} total, to {out}")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--company",
        action="append",
        dest="companies",
        help="company name to query (repeatable); default: every company sentiment.tickers can price",
    )
    parser.add_argument("--live", action="store_true", help="fetch from GDELT's live API instead of committed fixtures")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"CSV path to append parsed headlines to (default: the committed corpus with --live, {OFFLINE_OUT.name} in outputs/ offline)",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="with --live, do not overwrite the fixtures/gdelt/*.json snapshot on a successful fetch",
    )
    args = parser.parse_args()

    companies = args.companies if args.companies else known_companies()
    if not companies:
        print("no companies to query", file=sys.stderr)
        sys.exit(1)
    out = args.out if args.out is not None else (DEFAULT_OUT if args.live else OFFLINE_OUT)
    exit_code = run(companies, args.live, out, snapshot=not args.no_snapshot)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
