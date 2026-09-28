from datetime import datetime, timedelta, timezone
from pathlib import Path

from sentiment.rss import parse_feed_xml

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "rss" / "economic_times_markets.xml"
SCRAPED_AT = datetime(2026, 9, 28, 15, 43, 40, tzinfo=timezone.utc)

SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item>
  <title><![CDATA[Widget Corp beats earnings, raises guidance]]></title>
  <link>https://example.com/widget-corp-beats</link>
  <pubDate>Mon, 28 Sep 2026 08:26:07 +0530</pubDate>
</item>
<item>
  <title>Missing link item</title>
  <pubDate>Mon, 28 Sep 2026 08:26:07 +0530</pubDate>
</item>
<item>
  <title>Unparseable date item</title>
  <link>https://example.com/bad-date</link>
  <pubDate>not a real date</pubDate>
</item>
</channel></rss>
"""


def test_parses_committed_fixture():
    xml_bytes = FIXTURE.read_bytes()
    headlines = parse_feed_xml(xml_bytes, source="economic_times_markets", scraped_at=SCRAPED_AT)

    assert len(headlines) == 50
    first = headlines[0]
    assert first.source == "economic_times_markets"
    assert first.title
    assert first.link.startswith("https://economictimes.indiatimes.com/")
    assert first.published_at.tzinfo is not None
    assert first.scraped_at == SCRAPED_AT


def test_skips_malformed_items_but_keeps_well_formed_ones():
    headlines = parse_feed_xml(SAMPLE_XML, source="test_feed", scraped_at=SCRAPED_AT)

    assert len(headlines) == 1
    h = headlines[0]
    assert h.title == "Widget Corp beats earnings, raises guidance"
    assert h.link == "https://example.com/widget-corp-beats"


def test_preserves_published_timezone_offset():
    headlines = parse_feed_xml(SAMPLE_XML, source="test_feed", scraped_at=SCRAPED_AT)
    h = headlines[0]

    assert h.published_raw == "Mon, 28 Sep 2026 08:26:07 +0530"
    assert h.published_at.utcoffset() == timedelta(hours=5, minutes=30)
    # 08:26:07 IST is 02:56:07 UTC
    assert h.published_at.astimezone(timezone.utc).hour == 2
    assert h.published_at.astimezone(timezone.utc).minute == 56
