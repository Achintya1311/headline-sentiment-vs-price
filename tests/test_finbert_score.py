from datetime import datetime, timezone
from pathlib import Path

from sentiment.headline import Headline
from sentiment.finbert_score import (
    label_for_scores,
    read_scored_csv,
    score_headline,
    score_headlines,
    score_text,
    score_texts,
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


def test_score_text_returns_three_probabilities_summing_to_one():
    scores = score_text("Company posts record profit as sales surge")
    assert set(scores) == {"positive", "negative", "neutral"}
    assert abs(sum(scores.values()) - 1.0) < 1e-4


def test_clearly_positive_and_negative_headlines_get_the_expected_label():
    assert label_for_scores(score_text("Stock soars after strong quarterly results")) == "positive"
    assert label_for_scores(score_text("CEO resigns amid accounting scandal")) == "negative"


def test_finance_jargon_that_defeats_vader_is_correctly_read_by_finbert():
    # The exact two headlines Day 2's evaluation harness recorded as VADER
    # failures (see sentiment/vader_score.py and README Findings) - this is
    # the gap FinBERT exists to close.
    beats_guidance = score_text("Company beats earnings estimates, raises full-year guidance")
    assert label_for_scores(beats_guidance) == "positive"

    safety_recall = score_text("Manufacturer recalls product over safety defect")
    assert label_for_scores(safety_recall) == "negative"


def test_score_texts_batches_and_preserves_order():
    texts = ["Stock soars after strong quarterly results", "CEO resigns amid accounting scandal"]
    results = score_texts(texts)
    assert len(results) == 2
    assert label_for_scores(results[0]) == "positive"
    assert label_for_scores(results[1]) == "negative"


def test_score_texts_empty_list_returns_empty_list():
    assert score_texts([]) == []


def test_score_headline_wraps_the_source_headline_fields():
    h = make_headline("Stock soars after strong quarterly results")
    scored = score_headline(h)
    assert scored.source == h.source
    assert scored.title == h.title
    assert scored.link == h.link
    assert scored.label == "positive"
    assert scored.compound == scored.positive - scored.negative


def test_score_headlines_preserves_order():
    headlines = [make_headline("Stock soars", link="1"), make_headline("Stock plunges", link="2")]
    scored = score_headlines(headlines)
    assert [s.link for s in scored] == ["1", "2"]


def test_score_headlines_empty_list_returns_empty_list():
    assert score_headlines([]) == []


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
