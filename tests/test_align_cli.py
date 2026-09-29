import csv
from datetime import datetime, timezone
from pathlib import Path

from sentiment.align import run
from sentiment.headline import Headline, write_csv
from sentiment.market_hours import align_headline


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_run_aligns_every_headline_in_the_input_csv(tmp_path: Path):
    in_path = tmp_path / "raw.csv"
    out_path = tmp_path / "aligned.csv"
    write_csv(
        [
            make_headline("pre-open", "1", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
            make_headline("intraday", "2", datetime(2026, 9, 28, 5, 0, 0, tzinfo=timezone.utc)),
        ],
        in_path,
    )

    exit_code = run(in_path, out_path)

    assert exit_code == 0
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["timing"] for r in rows} == {"pre_open", "intraday"}
    for r in rows:
        assert r["source"] and r["title"] and r["link"] and r["published_at"] and r["local_time"] and r["session_date"]


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "aligned.csv"

    exit_code = run(missing, out_path)

    assert exit_code == 1
    assert not out_path.exists()


def test_run_against_committed_fixture_aligns_all_fifty_leak_free(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "aligned.csv"

    exit_code = run(fixture, out_path)

    assert exit_code == 0
    from sentiment.headline import read_csv

    headlines = read_csv(fixture)
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 50
    for h, row in zip(headlines, rows):
        alignment = align_headline(h.published_at)
        assert alignment.leak_free()
        assert row["session_date"] == alignment.session_date.isoformat()
        assert row["timing"] == alignment.timing.value

    timings = {row["timing"] for row in rows}
    # the real fixture spans pre-open, intraday and post-close headlines -
    # exercising all three branches against genuine scraped data, not just
    # constructed cases.
    assert timings == {"pre_open", "intraday", "post_close"}
