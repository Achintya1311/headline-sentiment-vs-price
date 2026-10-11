from datetime import datetime, timedelta, timezone

import pytest

from sentiment.headline import Headline
from sentiment.robustness import (
    DEFAULT_IN,
    names_price_move,
    run,
    session_demeaned_r,
    timestamp_integrity,
    within_session_permutation,
)

T0 = datetime(2026, 9, 28, 4, 0, tzinfo=timezone.utc)


def head(title, link, published, scraped=None):
    return Headline(
        source="feed", title=title, link=link, published_at=published,
        published_raw="x", scraped_at=scraped or published + timedelta(hours=1),
    )


def row(day, compound, ret):
    return {"session_date": day, "compound": compound, "contemporaneous_return": ret, "title": "t"}


@pytest.mark.parametrize("title", [
    "PC Jeweller shares rise 4% as firm becomes debt-free",
    "NSE shares fall 2%, slip below IPO price",
    "Great Eastern Shipping rises 3% as Nomura retains Buy",
    "Stock rallies 89% in 6 months",
])
def test_names_price_move_flags_headlines_that_report_a_move(title):
    assert names_price_move(title)


@pytest.mark.parametrize("title", [
    "Jefferies names Max Financial as its top insurance pick, sees 41% upside",
    "Infosys Share Price Highlights: Infosys Stock Price History",
    "HDFC Bank CEO transition nears; Jefferies sets target at Rs 1,200",
])
def test_names_price_move_ignores_forecasts_and_templates(title):
    assert not names_price_move(title)


def test_session_demeaned_r_removes_a_pure_market_wide_day_effect():
    # Day A everything +5%, day B everything -5%: the raw r is perfect only
    # because positive compounds sit on the up day. Within each day returns
    # are equal, so after demeaning there is nothing left to correlate.
    rows = [row("A", 0.8, 0.05), row("A", 0.7, 0.05), row("B", -0.8, -0.05), row("B", -0.7, -0.05)]
    assert session_demeaned_r(rows) == pytest.approx(0.0, abs=1e-9)


def test_session_demeaned_r_keeps_a_within_day_relationship():
    rows = [row("A", 0.9, 0.04), row("A", 0.1, 0.00), row("B", 0.8, 0.02), row("B", 0.2, -0.02)]
    assert session_demeaned_r(rows) > 0.9


def test_within_session_permutation_calls_a_day_effect_non_significant():
    # Same market-day confound: the real r looks perfect, but shuffling returns
    # inside each day reproduces it often, so p must be large.
    rows = [row("A", 0.8, 0.05), row("A", 0.7, 0.05), row("B", -0.8, -0.05), row("B", -0.7, -0.05)]
    res = within_session_permutation(rows, n_perm=500, seed=1)
    assert res.real_r > 0.99
    assert res.p_value > 0.5


def test_within_session_permutation_flags_a_real_within_day_signal():
    rows = [row("A", i / 10, i / 100) for i in range(10)] + [row("B", i / 10, i / 100) for i in range(10)]
    res = within_session_permutation(rows, n_perm=500, seed=1)
    assert res.real_r > 0.99
    assert res.p_value < 0.05


def test_within_session_permutation_never_reports_p_zero():
    rows = [row("A", i / 10, i / 100) for i in range(12)]
    assert within_session_permutation(rows, n_perm=50, seed=0).p_value >= 1 / 51


def test_timestamp_integrity_passes_clean_headlines():
    hs = [head("a", "l1", T0), head("b", "l2", T0 + timedelta(minutes=5))]
    assert timestamp_integrity(hs) == {}


def test_timestamp_integrity_flags_a_headline_dated_after_it_was_scraped():
    hs = [head("a", "l1", T0, scraped=T0 - timedelta(hours=1))]
    assert "published after it was scraped" in timestamp_integrity(hs)


def test_timestamp_integrity_flags_the_same_link_with_two_publish_times():
    hs = [head("a", "same", T0), head("a2", "same", T0 + timedelta(days=2))]
    assert "same link, different published time" in timestamp_integrity(hs)


def test_timestamp_integrity_flags_a_batch_stamped_instant():
    hs = [head(f"t{i}", f"l{i}", T0) for i in range(5)]
    assert any("exact timestamp" in k for k in timestamp_integrity(hs))


def test_run_on_the_committed_fixture_reports_and_exits_zero(capsys):
    assert run(DEFAULT_IN, n_perm=100, seed=0) == 0
    out = capsys.readouterr().out
    assert "6 of 23 headlines name a price move" in out
    assert "feed timestamps: all checks passed" in out
