from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentiment.headline import Headline, dedupe, merge_and_write, read_csv, write_csv


def make(source="feed", link="https://example.com/1", title="Title", minute=0):
    return Headline(
        source=source,
        title=title,
        link=link,
        published_at=datetime(2026, 9, 28, 8, minute, 0, tzinfo=timezone.utc),
        published_raw="Mon, 28 Sep 2026 08:00:00 +0000",
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        Headline(
            source="feed",
            title="t",
            link="https://example.com",
            published_at=datetime(2026, 9, 28, 8, 0, 0),  # naive
            published_raw="whatever",
            scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
        )


def test_dedupe_keeps_first_seen_same_source_and_link():
    a = make(link="https://example.com/1", title="First version")
    b = make(link="https://example.com/1", title="Rewritten headline")
    c = make(link="https://example.com/2", title="Different story")

    result = dedupe([a, b, c])

    assert result == [a, c]


def test_dedupe_treats_same_link_different_source_as_distinct():
    a = make(source="feed_a", link="https://example.com/1")
    b = make(source="feed_b", link="https://example.com/1")

    assert dedupe([a, b]) == [a, b]


def test_csv_round_trip(tmp_path: Path):
    headlines = [make(link="https://example.com/2", minute=5), make(link="https://example.com/1", minute=0)]
    out = tmp_path / "headlines.csv"

    write_csv(headlines, out)
    result = read_csv(out)

    # write_csv sorts by published_at ascending
    assert [h.link for h in result] == ["https://example.com/1", "https://example.com/2"]
    assert result[0].published_at == headlines[1].published_at
    assert result[0].scraped_at == headlines[1].scraped_at


def test_merge_and_write_accumulates_across_runs(tmp_path: Path):
    out = tmp_path / "headlines.csv"
    first_run = [make(link="https://example.com/1")]
    merge_and_write(out, first_run)

    # A later scrape no longer sees story 1 (outlet deleted/rewrote it) but
    # sees a new story 2 - both must survive on disk.
    second_run = [make(link="https://example.com/2")]
    merged = merge_and_write(out, second_run)

    links = {h.link for h in merged}
    assert links == {"https://example.com/1", "https://example.com/2"}
    assert {h.link for h in read_csv(out)} == links
