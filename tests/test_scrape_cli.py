from pathlib import Path

from sentiment.feeds import Feed
from sentiment.headline import read_csv
from sentiment.robots import RateLimiter, RobotsCache
from sentiment.scrape import run, scrape_feed_live

SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item>
  <title>Test headline</title>
  <link>https://example.com/story</link>
  <pubDate>Mon, 28 Sep 2026 08:26:07 +0530</pubDate>
</item>
</channel></rss>
"""


def test_offline_run_writes_expected_headline_count(tmp_path: Path):
    out = tmp_path / "headlines.csv"
    exit_code = run(["economic_times_markets"], live=False, out=out)

    assert exit_code == 0
    headlines = read_csv(out)
    assert len(headlines) == 50
    assert {h.source for h in headlines} == {"economic_times_markets"}


def test_offline_run_missing_fixture_reports_failure(tmp_path: Path):
    out = tmp_path / "headlines.csv"
    exit_code = run(["moneycontrol_business"], live=False, out=out)

    assert exit_code == 1
    assert not out.exists() or read_csv(out) == []


def test_live_success_snapshots_fixture(tmp_path: Path, monkeypatch):
    fixture_path = tmp_path / "fixtures" / "rss" / "fake_feed.xml"
    feed = Feed(name="fake_feed", url="https://fake.example/feed.xml", fixture=fixture_path)

    monkeypatch.setattr("sentiment.scrape.polite_get", lambda url, robots, limiter: SAMPLE_XML)

    headlines, status = scrape_feed_live(feed, RobotsCache(), RateLimiter())

    assert status == "ok"
    assert len(headlines) == 1
    assert fixture_path.read_bytes() == SAMPLE_XML


def test_live_success_no_snapshot_leaves_fixture_untouched(tmp_path: Path, monkeypatch):
    fixture_path = tmp_path / "fixtures" / "rss" / "fake_feed.xml"
    feed = Feed(name="fake_feed", url="https://fake.example/feed.xml", fixture=fixture_path)

    monkeypatch.setattr("sentiment.scrape.polite_get", lambda url, robots, limiter: SAMPLE_XML)

    headlines, status = scrape_feed_live(feed, RobotsCache(), RateLimiter(), snapshot=False)

    assert status == "ok"
    assert len(headlines) == 1
    assert not fixture_path.exists()
