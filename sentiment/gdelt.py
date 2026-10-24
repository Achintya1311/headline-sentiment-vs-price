"""GDELT Project DOC 2.0 API as a second, free headline source (Day 84, hub
CRITIQUES.md R18/R18's Day 84 queue).

Zero spend, no API key (PLAN.md rule 2/the hub's "no API keys" rule): the
GDELT Project states its datasets and API output are "available for
unlimited and unrestricted use" including automated/research use, the only
condition being attribution to the project and a link to gdeltproject.org
(https://www.gdeltproject.org/about.html, checked 24 Oct 2026) - unlike
Google News' own RSS feed, whose copyright notice restricts use to
"rendering... within a personal feed reader for personal, non-commercial
use" (checked the same day; see README's "Sources" paragraph for why that
source was considered and not added).

The live API itself rate-limits aggressively (observed: a 429 telling
callers to wait 5 seconds between requests, and a second 429 even after
waiting exactly that long) - ``DEFAULT_MIN_INTERVAL_SECONDS`` here is
deliberately above the API's own stated floor, and ``fetch_gdelt`` reuses
``sentiment.robots``'s ``RobotsCache``/``RateLimiter`` the same way every
other live fetch in this project does, so a denied or unreachable query is
skipped rather than retried into another 429.

A GDELT article's own ``domain`` field (e.g. ``"moneycontrol.com"``) becomes
each resulting ``Headline.source`` - not a blanket ``"gdelt"`` label - so
one GDELT query naturally contributes several distinct sources toward the
pre-registered test's ``>=3 sources`` gate (``sentiment.prereg``), the same
diversity the gate exists to require.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from sentiment.headline import Headline
from sentiment.robots import RateLimiter, RobotsCache, RobotsDenied

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# Observed: the API's own 429 asks for 5s between requests, and still 429'd
# once after exactly 5s in practice - stay well clear of its stated floor.
DEFAULT_MIN_INTERVAL_SECONDS = 15.0


def build_query_url(query: str, maxrecords: int = 50, sourcecountry: str = "India", sourcelang: str = "english") -> str:
    """A GDELT DOC 2.0 ``artlist``/JSON query URL for ``query``, scoped to
    English-language, India-sourced coverage - the same scope this
    project's headline resolution (``sentiment.tickers``) and FinBERT
    scoring (English-only model) already assume."""
    terms = query
    if sourcecountry:
        terms += f" sourcecountry:{sourcecountry}"
    if sourcelang:
        terms += f" sourcelang:{sourcelang}"
    params = {
        "query": terms,
        "mode": "artlist",
        "maxrecords": str(maxrecords),
        "format": "json",
        "sort": "datedesc",
    }
    return f"{GDELT_DOC_URL}?{urllib.parse.urlencode(params)}"


def _parse_seendate(raw: str) -> datetime:
    """GDELT's ``seendate`` is ``YYYYMMDDTHHMMSSZ`` (always UTC)."""
    return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def parse_gdelt_response(data: dict, scraped_at: datetime, language: str = "English") -> list[Headline]:
    """Return every article in a GDELT DOC API JSON response as a
    ``Headline``, keeping only ``language`` (default English, to match
    this project's scorers) and skipping any article missing a title, url
    or parseable ``seendate`` - the same "skip malformed, keep the rest"
    convention ``sentiment.rss.parse_feed_xml`` already uses. ``source`` is
    each article's own ``domain`` (see module docstring), falling back to
    ``"gdelt"`` only if a response ever omits it."""
    headlines: list[Headline] = []
    for article in data.get("articles", []):
        if language is not None and article.get("language") != language:
            continue
        title = (article.get("title") or "").strip()
        link = (article.get("url") or "").strip()
        raw_date = article.get("seendate") or ""
        if not title or not link or not raw_date:
            continue
        try:
            published_at = _parse_seendate(raw_date)
        except ValueError:
            continue
        headlines.append(
            Headline(
                source=article.get("domain") or "gdelt",
                title=title,
                link=link,
                published_at=published_at,
                published_raw=raw_date,
                scraped_at=scraped_at,
            )
        )
    return headlines


def fetch_gdelt_live(
    query: str,
    robots: RobotsCache,
    limiter: RateLimiter,
    maxrecords: int = 50,
    timeout: float = 15.0,
) -> tuple[list[Headline], str]:
    """Returns (headlines, status); status is "ok", "robots_denied", or
    "fetch_failed: <reason>" - the same contract ``sentiment.scrape``'s
    ``scrape_feed_live`` already uses, so callers can handle both sources
    the same way."""
    url = build_query_url(query, maxrecords=maxrecords)
    if not robots.can_fetch(url):
        return [], "robots_denied"
    host = urllib.parse.urlparse(url).netloc
    limiter.wait(host, extra_delay=max(robots.crawl_delay(url), DEFAULT_MIN_INTERVAL_SECONDS))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": robots.user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return [], f"fetch_failed: {e}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # The 429 rate-limit response is plain text, not JSON - treat the
        # same as any other fetch failure rather than crashing the run.
        return [], f"fetch_failed: non-JSON response ({e})"
    scraped_at = datetime.now(timezone.utc)
    return parse_gdelt_response(data, scraped_at=scraped_at), "ok"
