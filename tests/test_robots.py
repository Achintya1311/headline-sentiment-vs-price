from urllib.robotparser import RobotFileParser

from sentiment.robots import RateLimiter, RobotsCache

ROBOTS_TXT = """
User-agent: *
Disallow: /private/
Crawl-delay: 7

User-agent: headline-sentiment-vs-price-bot/0.1 (research project; contact gyaanvitaan@gmail.com)
Disallow: /also-blocked-for-us/
""".strip().splitlines()


def test_robotfileparser_semantics_disallow_and_allow():
    # sentiment.robots.RobotsCache is a thin cache around RobotFileParser;
    # this test pins the stdlib parsing behaviour it relies on, using an
    # in-memory robots.txt (no network) via RobotFileParser.parse().
    parser = RobotFileParser()
    parser.parse(ROBOTS_TXT)

    assert parser.can_fetch("some-other-bot", "https://example.com/private/x") is False
    assert parser.can_fetch("some-other-bot", "https://example.com/public/x") is True
    assert parser.crawl_delay("some-other-bot") == 7


def test_robots_cache_can_fetch_uses_cached_parser(monkeypatch):
    cache = RobotsCache(user_agent="test-agent")
    parser = RobotFileParser()
    parser.parse(ROBOTS_TXT)
    cache._parsers["example.com"] = parser  # avoid a real network fetch

    assert cache.can_fetch("https://example.com/public/x") is True
    assert cache.can_fetch("https://example.com/private/x") is False
    assert cache.crawl_delay("https://example.com/public/x") == 7


def test_robots_cache_fails_closed_on_403(monkeypatch):
    import urllib.error

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError("https://blocked.example/robots.txt", 403, "Forbidden", {}, None)

    monkeypatch.setattr("sentiment.robots.urllib.request.urlopen", fake_urlopen)

    cache = RobotsCache()
    assert cache.can_fetch("https://blocked.example/feed.xml") is False


def test_robots_cache_allows_all_on_missing_robots_txt(monkeypatch):
    import urllib.error

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError("https://no-robots.example/robots.txt", 404, "Not Found", {}, None)

    monkeypatch.setattr("sentiment.robots.urllib.request.urlopen", fake_urlopen)

    cache = RobotsCache()
    assert cache.can_fetch("https://no-robots.example/feed.xml") is True


def test_robots_cache_fails_closed_on_connection_error(monkeypatch):
    import urllib.error

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("sentiment.robots.urllib.request.urlopen", fake_urlopen)

    cache = RobotsCache()
    assert cache.can_fetch("https://unreachable.example/feed.xml") is False


def test_rate_limiter_enforces_minimum_gap(monkeypatch):
    fake_time = [1000.0]
    slept = []

    def fake_monotonic():
        return fake_time[0]

    def fake_sleep(seconds):
        slept.append(seconds)
        fake_time[0] += seconds

    monkeypatch.setattr("sentiment.robots.time.monotonic", fake_monotonic)
    monkeypatch.setattr("sentiment.robots.time.sleep", fake_sleep)

    limiter = RateLimiter(min_interval=5.0)
    limiter.wait("example.com")
    assert slept == []  # first call to a host never waits

    fake_time[0] += 1.0  # only 1s elapsed, need 5s
    limiter.wait("example.com")
    assert slept == [4.0]

    limiter.wait("other-host.com")
    assert slept == [4.0]  # a different host is not rate-limited by the first


def test_rate_limiter_respects_crawl_delay_over_default(monkeypatch):
    fake_time = [2000.0]
    slept = []

    monkeypatch.setattr("sentiment.robots.time.monotonic", lambda: fake_time[0])

    def fake_sleep(seconds):
        slept.append(seconds)
        fake_time[0] += seconds

    monkeypatch.setattr("sentiment.robots.time.sleep", fake_sleep)

    limiter = RateLimiter(min_interval=1.0)
    limiter.wait("example.com")
    fake_time[0] += 1.0
    limiter.wait("example.com", extra_delay=10.0)
    assert slept == [9.0]
