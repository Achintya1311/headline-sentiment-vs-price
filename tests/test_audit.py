import datetime as dt
import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    LEAK_MAX_MEAN_ABS_R,
    POWER_MAX_MEAN_ABS_R_AFTER,
    POWER_MIN_R_BEFORE,
    PowerCheckResult,
    SyntheticRow,
    build_synthetic_rows,
    correlation_r,
    leakage_check,
    power_check,
    run,
    shuffle_synthetic_pairing,
    shuffle_timestamps,
    synthetic_r,
)
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


# --- shuffle_timestamps -----------------------------------------------------


def test_shuffle_timestamps_preserves_multiset_and_titles():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert [h.title for h in shuffled] == ["a", "b", "c"]  # same headlines, same order
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)


def test_shuffle_timestamps_actually_reassigns_at_least_once_over_many_seeds():
    # A seed that happens to produce the identity permutation would make this
    # test vacuous - over many seeds, at least one must actually move a
    # timestamp to a different headline.
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]
    moved = False
    for seed in range(10):
        shuffled = shuffle_timestamps(headlines, random.Random(seed))
        if [h.published_at for h in shuffled] != [h.published_at for h in headlines]:
            moved = True
            break
    assert moved


def test_shuffle_timestamps_changes_which_session_a_headline_pairs_with(tmp_path: Path, monkeypatch):
    """Mechanical, non-statistical check: swapping two headlines' timestamps
    swaps which trading session's return each one's compound score is
    correlated against - the actual effect the leakage check relies on."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0), Bar(date=dt.date(2026, 9, 29), open=110.0, close=90.0)],
    )

    # Both headlines resolve to INFY.NS; pre-open on the 28th aligns to the
    # 28th, pre-open on the 29th aligns to the 29th.
    h_28 = make_headline(
        "Infosys Share Price Highlights: Infosys Stock Price History",
        "1",
        datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
    )
    h_29 = make_headline(
        "Infosys Share Price Highlights: Infosys Stock Price History",
        "2",
        datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
    )

    from sentiment.correlate import build_rows_from_headlines

    before, _ = build_rows_from_headlines([h_28, h_29])
    assert before[0]["session_date"] == "2026-09-28"
    assert before[0]["contemporaneous_return"] == (110.0 - 100.0) / 100.0
    assert before[1]["session_date"] == "2026-09-29"
    assert before[1]["contemporaneous_return"] == (90.0 - 110.0) / 110.0

    # Manually swap the two timestamps (what shuffle_timestamps does, made
    # deterministic here rather than relying on a seed landing on the swap).
    swapped = [
        Headline(**{**h_28.__dict__, "published_at": h_29.published_at}),
        Headline(**{**h_29.__dict__, "published_at": h_28.published_at}),
    ]
    after, _ = build_rows_from_headlines(swapped)
    assert after[0]["session_date"] == "2026-09-29"
    assert after[0]["contemporaneous_return"] == (90.0 - 110.0) / 110.0
    assert after[1]["session_date"] == "2026-09-28"
    assert after[1]["contemporaneous_return"] == (110.0 - 100.0) / 100.0


# --- correlation_r / leakage_check ------------------------------------------


def test_correlation_r_is_none_below_two_resolved_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]
    assert correlation_r(headlines) is None


def test_leakage_check_reports_none_when_nothing_resolves(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = [
        make_headline("Cyient among 4 stocks showing White Marubozu Pattern", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc))
    ]
    result = leakage_check(headlines, n_shuffles=20, seed=0)
    assert result.real_r is None
    assert result.n_resolved == 0
    assert result.passed is False


def test_leakage_check_mean_abs_shuffled_r_is_a_valid_correlation_magnitude(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=dt.date(2026, 9, 29), open=105.0, close=108.0),
            Bar(date=dt.date(2026, 9, 30), open=108.0, close=104.0),
        ],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "2",
            datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "3",
            datetime(2026, 9, 30, 2, 0, 0, tzinfo=timezone.utc),
        ),
    ]
    result = leakage_check(headlines, n_shuffles=50, seed=0)
    assert result.real_r is not None
    assert result.mean_abs_shuffled_r is not None
    assert 0.0 <= result.mean_abs_shuffled_r <= 1.0


# --- synthetic power check --------------------------------------------------


def test_build_synthetic_rows_from_fixtures(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=101.0),  # +1%
            Bar(date=dt.date(2026, 9, 29), open=100.0, close=99.0),  # -1%
        ],
    )
    rows = build_synthetic_rows(signal_gain=10.0)
    assert len(rows) == 2
    returns = sorted(r.contemporaneous_return for r in rows)
    assert returns[0] < 0 < returns[1]
    # compound = clip(10 * return): a +1% return -> compound 0.10, a -1% -> -0.10
    for row in rows:
        assert row.compound == max(-1.0, min(1.0, 10.0 * row.contemporaneous_return))


def test_synthetic_signal_is_strongly_correlated_before_shuffling(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "AAA.NS",
        [
            Bar(date=dt.date(2026, 9, d), open=100.0, close=100.0 * (1 + 0.01 * (d - 28)))
            for d in range(28, 31)
        ],
    )
    save_fixture(
        "BBB.NS",
        [
            Bar(date=dt.date(2026, 10, d), open=100.0, close=100.0 * (1 - 0.02 * d))
            for d in range(1, 4)
        ],
    )
    rows = build_synthetic_rows(signal_gain=10.0)
    r = synthetic_r(rows)
    assert r is not None
    assert r > 0.9  # compound is an exact (clipped) linear function of return


def test_shuffle_synthetic_pairing_preserves_marginals():
    rows = [
        SyntheticRow(ticker="A", session_date="2026-09-28", compound=0.5, contemporaneous_return=0.01),
        SyntheticRow(ticker="B", session_date="2026-09-29", compound=-0.3, contemporaneous_return=-0.02),
        SyntheticRow(ticker="C", session_date="2026-09-30", compound=0.1, contemporaneous_return=0.00),
    ]
    shuffled = shuffle_synthetic_pairing(rows, random.Random(0))
    assert sorted(r.compound for r in shuffled) == sorted(r.compound for r in rows)
    assert sorted((r.ticker, r.session_date, r.contemporaneous_return) for r in shuffled) == sorted(
        (r.ticker, r.session_date, r.contemporaneous_return) for r in rows
    )


def test_shuffle_synthetic_pairing_collapses_a_perfect_signal_on_average():
    # A clean, larger synthetic set with an exact linear signal (no clipping)
    # so the "before" correlation is exactly 1.0, making the post-shuffle
    # collapse unambiguous rather than a product of clipping noise.
    rows = [
        SyntheticRow(ticker=f"T{i}", session_date=f"2026-09-{i:02d}", compound=float(i), contemporaneous_return=float(i))
        for i in range(1, 21)
    ]
    assert synthetic_r(rows) == 1.0

    mean_abs_after = []
    for seed in range(30):
        shuffled = shuffle_synthetic_pairing(rows, random.Random(seed))
        r = synthetic_r(shuffled)
        assert r is not None
        mean_abs_after.append(abs(r))
    assert sum(mean_abs_after) / len(mean_abs_after) < 0.3


def test_power_check_passes_on_real_committed_fixtures():
    # Uses the repo's own committed fixtures/prices/*.json - no monkeypatch -
    # exercising the exact thresholds the CLI judges a real run against.
    result = power_check(n_shuffles=200, seed=0)
    assert isinstance(result, PowerCheckResult)
    assert result.r_before is not None
    assert abs(result.r_before) >= POWER_MIN_R_BEFORE
    assert result.mean_abs_r_after is not None
    assert result.mean_abs_r_after <= POWER_MAX_MEAN_ABS_R_AFTER
    assert result.passed


# --- CLI run() ---------------------------------------------------------------


def test_run_returns_zero_on_missing_input(tmp_path: Path):
    missing = tmp_path / "nope.csv"
    assert run(missing, n_shuffles=10, seed=0, live=False) == 1


def test_run_against_real_committed_fixture_passes():
    from sentiment.correlate import DEFAULT_IN

    exit_code = run(DEFAULT_IN, n_shuffles=100, seed=0, live=False)
    assert exit_code == 0


def test_leakage_check_fails_when_shuffling_does_not_change_the_correlation(tmp_path: Path, monkeypatch):
    """The dangerous bug this check exists to catch: a pipeline that ignores
    ``published_at`` (so shuffling it is a no-op) would still show whatever
    correlation the data has, shuffled or not. Simulated by monkeypatching
    ``correlation_r`` to always return the same strong value regardless of
    its input - a stand-in for "the shuffle changed nothing" - and
    confirming the check reports FAIL (high mean |shuffled r|), not a
    silent pass."""
    import sentiment.audit as audit_module

    headlines = [
        make_headline("x", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("x", "2", datetime(2026, 9, 29, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("x", "3", datetime(2026, 9, 30, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("x", "4", datetime(2026, 10, 1, 1, 0, 0, tzinfo=timezone.utc)),
    ]

    def fake_correlation_r(hs, live=False):
        return 0.97

    monkeypatch.setattr(audit_module, "correlation_r", fake_correlation_r)
    result = audit_module.leakage_check(headlines, n_shuffles=20, seed=0)
    assert result.mean_abs_shuffled_r is not None
    assert result.mean_abs_shuffled_r > LEAK_MAX_MEAN_ABS_R
    assert result.passed is False


def test_leakage_check_passes_when_shuffling_destroys_the_correlation(tmp_path: Path, monkeypatch):
    """The matching positive case: a shuffle mechanic that genuinely
    decorrelates compound from return should report PASS."""
    import sentiment.audit as audit_module

    headlines = [
        make_headline("x", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("x", "2", datetime(2026, 9, 29, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("x", "3", datetime(2026, 9, 30, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("x", "4", datetime(2026, 10, 1, 1, 0, 0, tzinfo=timezone.utc)),
    ]

    def fake_correlation_r(hs, live=False):
        return 0.0

    monkeypatch.setattr(audit_module, "correlation_r", fake_correlation_r)
    result = audit_module.leakage_check(headlines, n_shuffles=20, seed=0)
    assert result.mean_abs_shuffled_r is not None
    assert result.mean_abs_shuffled_r <= LEAK_MAX_MEAN_ABS_R
    assert result.passed is True
