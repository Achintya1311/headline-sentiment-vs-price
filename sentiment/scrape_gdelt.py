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

    python -m sentiment.scrape_gdelt
    python -m sentiment.scrape_gdelt --live
    python -m sentiment.scrape_gdelt --company Infosys --live
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sentiment.gdelt import fetch_gdelt_live, parse_gdelt_response
from sentiment.headline import Headline, merge_and_write, read_csv
from sentiment.robots import RateLimiter, RobotsCache
from sentiment.tickers import known_companies

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "gdelt"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


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

    for company in companies:
        if live:
            headlines, status = fetch_gdelt_live(company, robots, limiter)
            if status != "ok":
                print(f"[{company}] SKIPPED ({status})", file=sys.stderr)
                if status == "robots_denied":
                    continue
                exit_code = 1
                continue
            if snapshot:
                path = fixture_path(company)
                path.parent.mkdir(parents=True, exist_ok=True)
                # Re-fetch raw JSON for the snapshot rather than
                # re-serializing parsed Headlines, so the committed fixture
                # stays a faithful "trimmed real payload" (project
                # convention, see e.g. implied-vol-surface's raw captures).
                import urllib.request

                from sentiment.gdelt import build_query_url

                req = urllib.request.Request(
                    build_query_url(company), headers={"User-Agent": robots.user_agent}
                )
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    path.write_bytes(resp.read())
        else:
            try:
                headlines = scrape_company_offline(company)
                status = "ok (fixture)"
            except FileNotFoundError as e:
                print(f"[{company}] SKIPPED (no fixture: {e})", file=sys.stderr)
                continue
        print(f"[{company}] {status}: {len(headlines)} headlines")
        all_headlines.extend(headlines)

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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV path to append parsed headlines to")
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
    exit_code = run(companies, args.live, args.out, snapshot=not args.no_snapshot)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
