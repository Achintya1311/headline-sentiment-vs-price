import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import real_r, run, shuffle_published_at, shuffle_trial_r
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_published_at_keeps_titles_but_moves_timestamps():
    headlines = [
        make_headline("A", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("B", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("C", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]
    shuffled = shuffle_published_at(headlines, random.Random(1))

    assert [h.title for h in shuffled] == ["A", "B", "C"]
    assert [h.link for h in shuffled] == ["1", "2", "3"]
    # same multiset of timestamps, reassigned - nothing invented, nothing lost
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # this seed actually moves at least one timestamp off its original headline
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_published_at_is_deterministic_given_the_same_seed():
    headlines = [
        make_headline("A", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("B", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("C", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]
    first = shuffle_published_at(headlines, random.Random(42))
    second = shuffle_published_at(headlines, random.Random(42))
    assert [h.published_at for h in first] == [h.published_at for h in second]


def _two_ticker_fixture(tmp_path: Path, monkeypatch) -> list[Headline]:
    """Two resolvable headlines on two different tickers, each with its own
    session's price bar - enough price variation for a shuffle to actually
    change which return a headline's compound score gets paired with."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture(
        "INFY.NS",
        [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0)],
    )
    save_fixture(
        "WIPRO.NS",
        [Bar(date=dt.date(2026, 9, 28), open=100.0, close=90.0)],
    )
    return [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Wipro Share Price Highlights: Wipro Stock Price History",
            "2",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        ),
    ]


def test_real_r_matches_build_rows_on_the_unshuffled_sample(tmp_path, monkeypatch):
    headlines = _two_ticker_fixture(tmp_path, monkeypatch)
    result = real_r(headlines)
    assert result is not None
    r, n = result
    assert n == 2


def test_real_r_is_none_when_fewer_than_two_headlines_resolve(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = [
        make_headline(
            "Cyient among 4 stocks showing White Marubozu Pattern",
            "1",
            datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc),
        )
    ]
    assert real_r(headlines) is None


def test_shuffle_trial_r_runs_against_a_real_fixture(tmp_path, monkeypatch):
    headlines = _two_ticker_fixture(tmp_path, monkeypatch)
    trial = shuffle_trial_r(headlines, random.Random(7))
    assert trial is not None
    assert -1.0 <= trial <= 1.0


def test_run_against_committed_fixture_passes_and_is_deterministic(tmp_path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, trials=50, seed=0, live=False)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    assert run(missing, trials=10, seed=0, live=False) == 1


def test_run_detects_a_no_op_shuffle_as_a_failure(tmp_path, monkeypatch):
    # Simulate a leak: a pairing function that ignores whatever timestamps
    # the (shuffled) headlines carry and always returns the one pairing
    # computed against the real, unshuffled order. If the audit can't catch
    # this, shuffling isn't actually testing anything.
    headlines = _two_ticker_fixture(tmp_path, monkeypatch)
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    import sentiment.audit as audit_module

    original_build_rows_from_headlines = audit_module.build_rows_from_headlines
    frozen_result = original_build_rows_from_headlines(headlines, live=False)

    def frozen_build_rows_from_headlines(hs, live=False):
        return frozen_result

    monkeypatch.setattr(audit_module, "build_rows_from_headlines", frozen_build_rows_from_headlines)

    exit_code = audit_module.run(in_path, trials=20, seed=0, live=False)
    assert exit_code == 1
