"""Day 1 CLI: scrape headlines from RSS feeds, respecting robots.txt and rate limits.

Offline by default - reads the committed snapshots under ``fixtures/rss/``,
so the default path never needs network. ``--live`` attempts a real fetch
per feed instead, through :mod:`sentiment.robots` (robots.txt check + rate
limiter) - a feed that robots.txt disallows or that the network can't reach
is reported as such and skipped, it does not stop the run.

    python -m sentiment.scrape
    python -m sentiment.scrape --live
    python -m sentiment.scrape --feed economic_times_markets --live
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from sentiment.feeds import FEEDS, Feed
from sentiment.headline import Headline, merge_and_write, read_csv
from sentiment.robots import RateLimiter, RobotsCache, RobotsDenied, polite_get
from sentiment.rss import parse_feed_xml

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


def scrape_feed_offline(feed: Feed) -> list[Headline]:
    if not feed.fixture.exists():
        raise FileNotFoundError(
            f"no committed fixture for feed {feed.name!r} at {feed.fixture}; "
            "run with --live to fetch one, or pick a different --feed"
        )
    xml_bytes = feed.fixture.read_bytes()
    # Offline runs are reproducible: attribute every headline to the moment
    # this run happened, not to whenever the fixture was first captured.
    scraped_at = datetime.now(timezone.utc)
    return parse_feed_xml(xml_bytes, source=feed.name, scraped_at=scraped_at)


def scrape_feed_live(
    feed: Feed, robots: RobotsCache, limiter: RateLimiter, snapshot: bool = True
) -> tuple[list[Headline], str]:
    """Returns (headlines, status). status is "ok", "robots_denied", or "fetch_failed: <reason>".

    On success, also overwrites ``feed.fixture`` with the bytes just fetched
    when ``snapshot`` is set - outlets delete and rewrite headlines, so the
    only reproducible record is whatever we snapshotted at scrape time.
    """
    try:
        xml_bytes = polite_get(feed.url, robots, limiter)
    except RobotsDenied:
        return [], "robots_denied"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return [], f"fetch_failed: {e}"
    if snapshot:
        feed.fixture.parent.mkdir(parents=True, exist_ok=True)
        feed.fixture.write_bytes(xml_bytes)
    scraped_at = datetime.now(timezone.utc)
    return parse_feed_xml(xml_bytes, source=feed.name, scraped_at=scraped_at), "ok"


def run(feed_names: list[str], live: bool, out: Path, snapshot: bool = True) -> int:
    robots = RobotsCache()
    limiter = RateLimiter()
    all_headlines: list[Headline] = []
    exit_code = 0

    for name in feed_names:
        feed = FEEDS[name]
        if live:
            headlines, status = scrape_feed_live(feed, robots, limiter, snapshot=snapshot)
            if status != "ok":
                print(f"[{feed.name}] SKIPPED ({status})", file=sys.stderr)
                if status == "robots_denied":
                    continue
                exit_code = 1
                continue
        else:
            try:
                headlines = scrape_feed_offline(feed)
                status = "ok (fixture)"
            except FileNotFoundError as e:
                print(f"[{feed.name}] SKIPPED (no fixture: {e})", file=sys.stderr)
                exit_code = 1
                continue
        print(f"[{feed.name}] {status}: {len(headlines)} headlines")
        all_headlines.extend(headlines)

    prior_count = len(read_csv(out)) if out.exists() else 0
    merged = merge_and_write(out, all_headlines)
    new_count = len(merged) - prior_count
    print(f"scraped {len(all_headlines)}, {new_count} new after dedup, {len(merged)} total, to {out}")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--feed",
        action="append",
        dest="feeds",
        choices=sorted(FEEDS),
        help="feed name to scrape (repeatable); default: all registered feeds",
    )
    parser.add_argument("--live", action="store_true", help="fetch from the network instead of committed fixtures")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV path to append parsed headlines to")
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="with --live, do not overwrite the fixtures/rss/*.xml snapshot on a successful fetch",
    )
    args = parser.parse_args()

    if args.feeds:
        feed_names = args.feeds
    elif args.live:
        feed_names = sorted(FEEDS)
    else:
        # Offline default: only feeds with a committed snapshot to read.
        # A feed registered but never fetched live (no fixture yet) is a
        # future --live candidate, not a failure of an unqualified run.
        feed_names = sorted(name for name, feed in FEEDS.items() if feed.fixture.exists())
        if not feed_names:
            print("no committed RSS fixtures found; run with --live first", file=sys.stderr)
            sys.exit(1)
    exit_code = run(feed_names, args.live, args.out, snapshot=not args.no_snapshot)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
