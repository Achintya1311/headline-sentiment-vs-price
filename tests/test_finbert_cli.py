from datetime import datetime, timezone
from pathlib import Path

from sentiment.headline import Headline, write_csv
from sentiment.finbert import run
from sentiment.finbert_score import read_scored_csv


def make_headline(title: str, link: str) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc),
        published_raw="Mon, 28 Sep 2026 08:00:00 +0000",
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_run_scores_every_headline_in_the_input_csv(tmp_path: Path):
    in_path = tmp_path / "raw.csv"
    out_path = tmp_path / "scored.csv"
    write_csv(
        [make_headline("Stock soars after strong quarterly results", "1"),
         make_headline("CEO resigns amid accounting scandal", "2")],
        in_path,
    )

    exit_code = run(in_path, out_path)

    assert exit_code == 0
    scored = read_scored_csv(out_path)
    assert len(scored) == 2
    assert {s.label for s in scored} == {"positive", "negative"}


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "scored.csv"

    exit_code = run(missing, out_path)

    assert exit_code == 1
    assert not out_path.exists()


def test_run_against_committed_fixture_scores_all_fifty(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "scored.csv"

    exit_code = run(fixture, out_path)

    assert exit_code == 0
    assert len(read_scored_csv(out_path)) == 50
