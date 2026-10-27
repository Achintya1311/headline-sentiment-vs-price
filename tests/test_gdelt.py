import json
from datetime import datetime, timezone
from pathlib import Path

from sentiment.gdelt import build_query_url, fetch_gdelt_live, parse_gdelt_response
from sentiment.robots import RateLimiter, RobotsCache, RobotsDenied

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "gdelt" / "infosys.json"
SCRAPED_AT = datetime(2026, 10, 24, 15, 0, 0, tzinfo=timezone.utc)


def test_build_query_url_includes_country_and_language_scoping():
    url = build_query_url("Infosys")
    assert "query=Infosys" in url or "query=Infosys+sourcecountry" in url
    assert "sourcecountry" in url
    assert "sourcelang" in url
    assert "format=json" in url


def test_parses_the_committed_real_sample_keeping_only_english():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    headlines = parse_gdelt_response(data, scraped_at=SCRAPED_AT)

    # The real sample mixes English and Hindi articles (seen in practice,
    # see sentiment/gdelt.py's module docstring) - only English survives.
    assert len(headlines) > 0
    assert all(h.title for h in headlines)
    assert "इंफोसिस" not in " ".join(h.title for h in headlines)


def test_source_is_the_article_domain_not_a_blanket_gdelt_label():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    headlines = parse_gdelt_response(data, scraped_at=SCRAPED_AT)

    sources = {h.source for h in headlines}
    assert len(sources) >= 2
    assert "gdelt" not in sources


def test_parse_skips_malformed_articles_but_keeps_well_formed_ones():
    data = {
        "articles": [
            {"title": "Good headline", "url": "https://example.com/1", "seendate": "20261024T101500Z", "domain": "example.com", "language": "English"},
            {"title": "", "url": "https://example.com/2", "seendate": "20261024T101500Z", "domain": "example.com", "language": "English"},
            {"title": "No link", "url": "", "seendate": "20261024T101500Z", "domain": "example.com", "language": "English"},
            {"title": "Bad date", "url": "https://example.com/4", "seendate": "not-a-date", "domain": "example.com", "language": "English"},
        ]
    }
    headlines = parse_gdelt_response(data, scraped_at=SCRAPED_AT)
    assert len(headlines) == 1
    assert headlines[0].title == "Good headline"


def test_fetch_gdelt_live_returns_robots_denied_without_network(monkeypatch):
    robots = RobotsCache()
    monkeypatch.setattr(robots, "can_fetch", lambda url: False)
    headlines, status = fetch_gdelt_live("Infosys", robots, RateLimiter())
    assert status == "robots_denied"
    assert headlines == []


def test_fetch_gdelt_live_reports_fetch_failed_on_a_network_error(monkeypatch):
    import urllib.error

    robots = RobotsCache()
    monkeypatch.setattr(robots, "can_fetch", lambda url: True)
    monkeypatch.setattr(robots, "crawl_delay", lambda url: 0.0)

    def boom(*a, **kw):
        raise urllib.error.URLError("no network in this sandbox")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    headlines, status = fetch_gdelt_live("Infosys", robots, RateLimiter(), timeout=0.1)
    assert status.startswith("fetch_failed")
    assert headlines == []


def test_fetch_gdelt_live_treats_the_rate_limit_text_response_as_a_fetch_failure(monkeypatch, tmp_path):
    robots = RobotsCache()
    monkeypatch.setattr(robots, "can_fetch", lambda url: True)
    monkeypatch.setattr(robots, "crawl_delay", lambda url: 0.0)

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"Please limit requests to one every 5 seconds..."

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: FakeResp())
    headlines, status = fetch_gdelt_live("Infosys", robots, RateLimiter())
    assert status.startswith("fetch_failed")
    assert headlines == []


def test_fetch_gdelt_live_says_unreachable_when_robots_could_not_be_read(monkeypatch):
    # R27: the Actions logs showed every company "robots_denied" after a
    # 10 s robots.txt timeout; that is an unreadable file, not a disallow.
    robots = RobotsCache()
    monkeypatch.setattr(robots, "can_fetch", lambda url: False)
    monkeypatch.setattr(robots, "unreadable_reason", lambda url: "<urlopen error timed out>")
    headlines, status = fetch_gdelt_live("Infosys", robots, RateLimiter())
    assert status == "robots_unreachable: <urlopen error timed out>"
    assert headlines == []


def test_scrape_gdelt_run_summarises_when_every_query_was_skipped(monkeypatch, tmp_path, capsys):
    from sentiment import scrape_gdelt

    monkeypatch.setattr(
        scrape_gdelt, "fetch_gdelt_live", lambda company, robots, limiter: ([], "robots_unreachable: timed out")
    )
    code = scrape_gdelt.run(["Infosys", "Wipro"], live=True, out=tmp_path / "h.csv", snapshot=False)
    err = capsys.readouterr().err
    assert code == 1
    assert "[Infosys] SKIPPED (robots_unreachable: timed out)" in err
    assert "GDELT: 0 of 2 queries fetched (2 robots_unreachable)" in err
