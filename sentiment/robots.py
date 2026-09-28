"""robots.txt permission checks and a polite, rate-limited fetch (Day 1).

Two things a scraper must not skip: asking robots.txt before every request,
and never hammering a host. Both live here so ``scrape.py`` cannot fetch a
URL without going through them.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

USER_AGENT = "headline-sentiment-vs-price-bot/0.1 (research project; contact gyaanvitaan@gmail.com)"
DEFAULT_MIN_INTERVAL_SECONDS = 3.0


class RobotsDenied(Exception):
    """Raised when robots.txt disallows fetching a URL for our user agent."""


class RateLimiter:
    """Enforces a minimum gap between requests to the same host.

    ``min_interval`` is a floor; a host's own ``Crawl-delay`` (read from
    robots.txt, when present) can only raise it, never lower it below this
    floor.
    """

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS) -> None:
        self.min_interval = min_interval
        self._last_request_at: dict[str, float] = {}

    def wait(self, host: str, extra_delay: float = 0.0) -> float:
        """Sleep as needed so this call is at least the required gap after the last one to ``host``.

        Returns the number of seconds actually slept (for tests/logging).
        """
        interval = max(self.min_interval, extra_delay)
        now = time.monotonic()
        last = self._last_request_at.get(host)
        slept = 0.0
        if last is not None:
            elapsed = now - last
            if elapsed < interval:
                slept = interval - elapsed
                time.sleep(slept)
        self._last_request_at[host] = time.monotonic()
        return slept


class RobotsCache:
    """Fetches and caches robots.txt per host, so each host is asked once per run."""

    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 10.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, RobotFileParser] = {}

    def _get_parser(self, host: str, scheme: str) -> RobotFileParser:
        if host in self._parsers:
            return self._parsers[host]
        robots_url = f"{scheme}://{host}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            req = urllib.request.Request(robots_url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                lines = resp.read().decode("utf-8", errors="replace").splitlines()
            parser.parse(lines)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                # Per RFC 9309: robots.txt itself refused -> treat as fully disallowed.
                parser.disallow_all = True
            else:
                # e.g. 404: no robots.txt published -> treat as fully allowed.
                parser.allow_all = True
        except (urllib.error.URLError, TimeoutError, OSError):
            # Can't confirm permission at all. Fail closed: do not scrape a
            # host whose robots.txt we could not read.
            parser.disallow_all = True
        self._parsers[host] = parser
        return parser

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        parser = self._get_parser(parsed.netloc, parsed.scheme)
        return parser.can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float:
        parsed = urlparse(url)
        parser = self._get_parser(parsed.netloc, parsed.scheme)
        delay = parser.crawl_delay(self.user_agent)
        return float(delay) if delay is not None else 0.0


def polite_get(
    url: str,
    robots: RobotsCache,
    limiter: RateLimiter,
    timeout: float = 10.0,
) -> bytes:
    """Fetch ``url``, but only after checking robots.txt and respecting the rate limit.

    Raises ``RobotsDenied`` rather than fetching when disallowed.
    """
    if not robots.can_fetch(url):
        raise RobotsDenied(f"robots.txt disallows fetching {url}")
    host = urlparse(url).netloc
    limiter.wait(host, extra_delay=robots.crawl_delay(url))
    req = urllib.request.Request(url, headers={"User-Agent": robots.user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()
