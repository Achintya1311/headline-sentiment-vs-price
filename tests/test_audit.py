import datetime as dt
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import run, run_leakage_audit, shuffle_timestamps
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


def test_shuffle_timestamps_preserves_the_exact_timestamp_multiset():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2 + i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    rng = random.Random(0)

    shuffled = shuffle_timestamps(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]  # titles never move
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]  # actually permuted


def test_shuffle_timestamps_is_deterministic_given_the_same_seed():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2 + i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]

    a = shuffle_timestamps(headlines, random.Random(42))
    b = shuffle_timestamps(headlines, random.Random(42))

    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_run_leakage_audit_against_real_fixture_matches_day5s_documented_null_result():
    # README Findings (Day 5): contemporaneous r=-0.185, n=23. The audit must
    # reproduce exactly that real-pairing statistic, and - the actual point
    # of this test - find it unremarkable against a shuffled-timestamp null:
    # Day 5/6 already concluded there is no detectable signal here, so the
    # leakage test on this fixture should pass, not flag a leak.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = run_leakage_audit(fixture, n_shuffles=200, seed=0)

    assert result.real_r == pytest.approx(-0.185, abs=0.001)
    assert result.real_n == 23
    assert result.n_shuffles_skipped == 0
    assert len(result.null_rs) == 200
    assert not result.leaks
    assert result.p_value >= 0.05


def test_run_leakage_audit_flags_a_real_correlation_that_does_not_survive_shuffling(monkeypatch, tmp_path: Path):
    # An engineered scenario with a real, strong timing-dependent signal:
    # each headline's VADER compound (monkeypatched, so it is exactly
    # controlled rather than depending on lexicon wording) lines up with the
    # one session its *correct* timestamp aligns to. The point: this kind of
    # too-good-to-be-true correlation on a single ticker is exactly the
    # shape a real leak (e.g. an alignment bug) would produce, so the audit
    # should treat the real pairing as a statistical outlier once timestamps
    # are randomly re-paired.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    n = 10
    trading_days = []
    day = dt.date(2026, 9, 1)
    while len(trading_days) < n:
        if day.weekday() < 5:
            trading_days.append(day)
        day += dt.timedelta(days=1)

    # strictly increasing session return, day over day
    bars = [Bar(date=d, open=100.0, close=100.0 * (1 + 0.01 * i)) for i, d in enumerate(trading_days)]
    save_fixture("TICK.NS", bars)

    compounds = {f"Headline {i}": (i - (n - 1) / 2) / n for i in range(n)}
    monkeypatch.setattr(correlate, "resolve", lambda title: ("Tick Co", "TICK.NS"))

    def fake_score_headline(headline):
        class _Scored:
            compound = compounds[headline.title]

        return _Scored()

    monkeypatch.setattr(correlate, "score_headline", fake_score_headline)

    # 02:00 UTC = 07:30 IST, before the 09:15 IST open - each headline is
    # pre-open on its own trading day, so align_headline assigns it to that
    # exact day's session (the "correct" pairing this test engineers).
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=timezone.utc))
        for i, d in enumerate(trading_days)
    ]
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    result = run_leakage_audit(in_path, n_shuffles=200, seed=0)

    assert result.real_r == pytest.approx(1.0, abs=1e-6)  # perfect by construction
    assert result.leaks
    assert result.p_value < 0.05
    assert result.null_mean_abs_r < result.real_r  # shuffling typically destroys it


def test_run_cli_reports_pass_against_committed_fixture(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=50, seed=0, live=False)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "real contemporaneous correlation: r=-0.185" in out
    assert "PASS" in out


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=50, seed=0, live=False)

    assert exit_code == 1


def test_run_leakage_audit_raises_when_too_few_rows_resolve(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Cyient among 4 stocks showing White Marubozu Pattern", "1", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )

    with pytest.raises(ValueError):
        run_leakage_audit(in_path, n_shuffles=10, seed=0)
