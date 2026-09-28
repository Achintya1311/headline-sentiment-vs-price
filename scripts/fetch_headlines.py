"""Refresh fixtures/rss/*.xml and append to fixtures/headlines/headlines_raw.csv (Day 1).

Not part of the runtime import path - sentiment.scrape reads committed
fixtures by default, so tests and the offline CLI never need network. This
script is how those fixtures get refreshed: a real, robots-and-rate-limit
respecting fetch against the live feeds in sentiment.feeds.FEEDS.

A feed robots.txt disallows, or that the sandbox's network can't reach, is
reported and skipped rather than failing the run - both are expected
outcomes here, not bugs (see README limitations).

Usage:
    python scripts/fetch_headlines.py
    python scripts/fetch_headlines.py --feed economic_times_markets
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentiment.scrape import main  # noqa: E402

if __name__ == "__main__":
    sys.argv.insert(1, "--live")
    main()
