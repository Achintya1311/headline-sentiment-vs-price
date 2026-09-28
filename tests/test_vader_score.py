from datetime import datetime, timezone
from pathlib import Path

from sentiment.headline import Headline
from sentiment.vader_score import (
    label_for_compound,
    read_scored_csv,
    score_headline,
    score_headlines,
    score_text,
    write_scored_csv,
)


def make_headline(title: str, link: str = "https://example.com/1") -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc),
        published_raw="Mon, 28 Sep 2026 08:00:00 +0000",
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_score_text_returns_the_four_vader_fields():
    scores = score_text("Company posts record profit as sales surge")
    assert set(scores) == {"neg", "neu", "pos", "compound"}
    assert scores["compound"] > 0


def test_clearly_positive_and_negative_headlines_get_the_expected_sign():
    assert score_text("Stock soars after strong quarterly results")["compound"] > 0
    assert score_text("CEO resigns amid accounting scandal")["compound"] < 0


def test_label_for_compound_thresholds():
    assert label_for_compound(0.05) == "positive"
    assert label_for_compound(0.049) == "neutral"
    assert label_for_compound(-0.05) == "negative"
    assert label_for_compound(-0.049) == "neutral"
    assert label_for_compound(0.0) == "neutral"


def test_finance_jargon_scores_as_neutral_not_positive():
    # The documented VADER caveat from NEXT_STEPS.md: bullish finance jargon
    # with no everyday sentiment words is invisible to a general lexicon.
    scores = score_text("Company beats earnings estimates, raises full-year guidance")
    assert scores["compound"] == 0.0
    assert label_for_compound(scores["compound"]) == "neutral"


def test_score_headline_wraps_the_source_headline_fields():
    h = make_headline("Stock soars after strong quarterly results")
    scored = score_headline(h)
    assert scored.source == h.source
    assert scored.title == h.title
    assert scored.link == h.link
    assert scored.label == "positive"


def test_score_headlines_preserves_order():
    headlines = [make_headline("Stock soars", link="1"), make_headline("Stock plunges", link="2")]
    scored = score_headlines(headlines)
    assert [s.link for s in scored] == ["1", "2"]


def test_scored_csv_round_trip(tmp_path: Path):
    headlines = [
        make_headline("Stock soars after strong quarterly results", link="1"),
        make_headline("CEO resigns amid accounting scandal", link="2"),
    ]
    scored = score_headlines(headlines)
    out = tmp_path / "scored.csv"

    write_scored_csv(scored, out)
    result = read_scored_csv(out)

    assert result == scored
