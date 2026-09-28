"""Parse RSS feed XML into raw ``Headline`` records (Day 1).

Pure parsing, no network. Feeds committed under ``fixtures/rss/`` are read
by this module directly, so tests and the offline CLI path never need a
live connection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from bs4 import BeautifulSoup

from sentiment.headline import Headline


def _parse_pubdate(raw: str) -> datetime:
    """RSS ``pubDate`` is RFC 2822 (e.g. ``Mon, 28 Sep 2026 20:47:16 +0530``).

    ``parsedate_to_datetime`` keeps whatever UTC offset the feed published,
    which is exactly what we want to preserve at scrape time - converting to
    a single timezone is a display choice, not a parsing one.
    """
    dt = parsedate_to_datetime(raw)
    if dt.tzinfo is None:
        # RFC 2822 allows a missing/"-0000" offset meaning "unknown", which
        # parsedate_to_datetime represents as naive. Treat unknown as UTC
        # rather than silently mislocating the headline in time.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_feed_xml(xml_bytes: bytes, source: str, scraped_at: datetime) -> list[Headline]:
    """Return every ``<item>`` in an RSS 2.0 document as a ``Headline``.

    Items missing a title, link, or parseable pubDate are skipped rather
    than raising - a malformed item in one story should not lose the rest
    of the feed. ``scraped_at`` should be timezone-aware UTC.
    """
    soup = BeautifulSoup(xml_bytes, "xml")
    headlines: list[Headline] = []
    for item in soup.find_all("item"):
        title_tag = item.find("title")
        link_tag = item.find("link")
        pubdate_tag = item.find("pubDate")
        if title_tag is None or link_tag is None or pubdate_tag is None:
            continue
        title = title_tag.get_text(strip=True)
        link = link_tag.get_text(strip=True)
        raw_date = pubdate_tag.get_text(strip=True)
        if not title or not link or not raw_date:
            continue
        try:
            published_at = _parse_pubdate(raw_date)
        except (TypeError, ValueError):
            continue
        headlines.append(
            Headline(
                source=source,
                title=title,
                link=link,
                published_at=published_at,
                published_raw=raw_date,
                scraped_at=scraped_at,
            )
        )
    return headlines
