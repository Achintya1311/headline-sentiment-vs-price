"""Registry of RSS feeds this project scrapes (Day 1).

One entry per feed: a short name (used as the ``source`` field on every
``Headline`` from it, and as the fixture filename stem), the live URL, and
the path to the committed offline snapshot under ``fixtures/rss/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "rss"


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    fixture: Path


FEEDS: dict[str, Feed] = {
    "economic_times_markets": Feed(
        name="economic_times_markets",
        url="https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        fixture=FIXTURES_DIR / "economic_times_markets.xml",
    ),
    "moneycontrol_business": Feed(
        name="moneycontrol_business",
        url="https://www.moneycontrol.com/rss/business.xml",
        fixture=FIXTURES_DIR / "moneycontrol_business.xml",
    ),
    "reuters_business": Feed(
        name="reuters_business",
        url="https://feeds.reuters.com/reuters/businessNews",
        fixture=FIXTURES_DIR / "reuters_business.xml",
    ),
}
