import datetime as dt
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    AuditResult,
    _mean_ci,
    permutation_p_value,
    run,
    run_audit,
    shuffle_timestamps,
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


def test_shuffle_timestamps_is_a_permutation_not_fresh_random_times():
    headlines = [
        make_headline(f"HDFC Bank shares move {i}%", f"link-{i}", datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(8)
    ]
    rng = random.Random(1)

    shuffled = shuffle_timestamps(headlines, rng)

    # every original timestamp still appears exactly once, just reassigned
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links are untouched - only the timestamp column moved
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # with 8 headlines and this seed, the permutation is not the identity
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_permutation_p_value_small_when_real_result_is_far_from_the_shuffled_pack():
    p = permutation_p_value(real_r=0.9, shuffled_rs=[0.05, -0.1, 0.02, -0.03, 0.08])
    assert p == pytest.approx(1 / 6)


def test_permutation_p_value_large_when_shuffled_runs_match_the_real_result():
    p = permutation_p_value(real_r=0.5, shuffled_rs=[0.6, 0.55, 0.52, 0.51])
    assert p == pytest.approx(1.0)


def test_permutation_p_value_requires_at_least_one_shuffle():
    with pytest.raises(ValueError):
        permutation_p_value(real_r=0.5, shuffled_rs=[])


def test_mean_ci_zero_variance_collapses_to_a_point():
    mean, lo, hi = _mean_ci([1.0, 1.0, 1.0])
    assert mean == lo == hi == 1.0


def test_mean_ci_widens_with_variance_and_narrows_with_sample_size():
    narrow = _mean_ci([0.0, 0.0, 0.0, 0.0])
    wide = _mean_ci([-1.0, 1.0, -1.0, 1.0])
    assert wide[2] - wide[1] > narrow[2] - narrow[1]

    many = _mean_ci([-1.0, 1.0] * 50)
    few = _mean_ci([-1.0, 1.0])
    assert many[2] - many[1] < few[2] - few[1]


def test_run_audit_against_the_committed_fixture_matches_the_documented_null(monkeypatch):
    # Day 5's README finding: contemporaneous r = -0.185, n = 23. The audit
    # must reproduce the same real_r (it reuses correlate's own pipeline)
    # and, since that result was already an honest null, the shuffled
    # control should straddle zero too - there is no signal here to leak.
    from sentiment.headline import read_csv

    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    result = run_audit(headlines, n_shuffles=100, seed=0, live=False)

    assert result.n_real_rows == 23
    assert result.real_r == pytest.approx(-0.185, abs=0.001)
    assert result.shuffled_ci_low <= 0.0 <= result.shuffled_ci_high
    assert not result.leak_suspected


def test_run_against_the_committed_fixture_passes():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=100, seed=0, live=False)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=50, seed=0, live=False)

    assert exit_code == 1


def test_run_audit_raises_when_fewer_than_two_headlines_resolve(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    with pytest.raises(ValueError):
        run_audit(headlines, n_shuffles=10, seed=0, live=False)


# --- the synthetic signal-disappears-under-shuffle test ---------------------
#
# This is the proof the gate has power, not just a pass on a pipeline that
# happens to already find nothing (see the committed-fixture test above,
# and the README's Day 8 findings): build a fixture with a real, engineered
# correlation that depends entirely on correct timestamp alignment, and
# confirm shuffling timestamps destroys it. Four headlines are timed
# pre-open on day 1 and carry clearly positive sentiment; day 1's return is
# +5%. Four more are timed post-close on day 1 (so Day 4's alignment rolls
# them to day 2) and carry clearly negative sentiment; day 2's return is
# -5%, the same for every ticker. Correctly aligned, sentiment sign matches
# the paired session's return sign perfectly (r = +1.0). Shuffling
# timestamps randomly reassigns which 4 of the 8 tickers land on day 1 vs
# day 2 - a company's own fixed sentiment no longer has anything to do with
# which session it gets paired with, so across many shuffles the average
# correlation should be indistinguishable from zero (a permutation-test
# null by construction, whatever the actual sentiment values are).

DAY_1 = dt.date(2026, 9, 28)  # Monday - a trading day
DAY_2 = dt.date(2026, 9, 29)  # Tuesday - the next trading day

POSITIVE_HEADLINES = [
    ("PC Jeweller shares soar on excellent profit and strong growth", "PCJEWELLER.NS"),
    ("Fortis Healthcare shares rally on great earnings and positive outlook", "FORTIS.NS"),
    ("BSE shares jump on excellent trading volumes and strong growth", "BSE.NS"),
    ("PB Fintech shares surge on great guidance and strong profit", "POLICYBZR.NS"),
]
NEGATIVE_HEADLINES = [
    ("NSE shares crash amid terrible losses and investor panic", "NSE.BO"),
    ("Suzlon Energy shares tumble as bad debt fears spark panic selling", "SUZLON.NS"),
    ("HDFC Bank shares tumble as bad debt fears spark panic selling", "HDFCBANK.NS"),
    ("Great Eastern Shipping shares crash amid terrible losses and investor panic", "GESHIP.NS"),
]
PRE_OPEN_UTC = datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)  # 07:30 IST, before the 09:15 open
POST_CLOSE_UTC = datetime(2026, 9, 28, 11, 0, 0, tzinfo=timezone.utc)  # 16:30 IST, after the 15:30 close


def _build_engineered_headlines() -> list[Headline]:
    headlines = []
    for i, (title, _ticker) in enumerate(POSITIVE_HEADLINES):
        headlines.append(make_headline(title, f"pos-{i}", PRE_OPEN_UTC))
    for i, (title, _ticker) in enumerate(NEGATIVE_HEADLINES):
        headlines.append(make_headline(title, f"neg-{i}", POST_CLOSE_UTC))
    return headlines


def test_engineered_headlines_actually_resolve_and_align_as_designed():
    # Guards the test fixture itself: if sentiment.tickers or
    # sentiment.market_hours ever change shape, this fails loudly here
    # instead of the power test below silently testing nothing.
    from sentiment.market_hours import align_headline
    from sentiment.tickers import resolve

    for title, ticker in POSITIVE_HEADLINES:
        company, resolved_ticker = resolve(title)
        assert resolved_ticker == ticker, title
        assert align_headline(PRE_OPEN_UTC).session_date == DAY_1

    for title, ticker in NEGATIVE_HEADLINES:
        company, resolved_ticker = resolve(title)
        assert resolved_ticker == ticker, title
        assert align_headline(POST_CLOSE_UTC).session_date == DAY_2


def test_a_real_timing_dependent_signal_disappears_under_timestamp_shuffling(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    all_tickers = [t for _, t in POSITIVE_HEADLINES] + [t for _, t in NEGATIVE_HEADLINES]
    for ticker in all_tickers:
        save_fixture(
            ticker,
            [
                Bar(date=DAY_1, open=100.0, close=105.0),  # +5%
                Bar(date=DAY_2, open=100.0, close=95.0),  # -5%
            ],
        )

    headlines = _build_engineered_headlines()

    result = run_audit(headlines, n_shuffles=300, seed=42, live=False)

    # Correctly aligned: positive sentiment pairs with day 1's +5%, negative
    # sentiment pairs with day 2's -5% - a clean, strong real correlation.
    assert result.real_r > 0.6
    # Shuffling which headline gets which publish time destroys that pairing:
    # the null clusters around zero and its 95% CI should contain it, unlike
    # the real, unshuffled result.
    assert result.shuffled_ci_low <= 0.0 <= result.shuffled_ci_high
    assert abs(result.shuffled_mean) < 0.3
    assert not result.leak_suspected


def test_a_correlation_that_ignores_timestamps_entirely_is_flagged_as_leaking(tmp_path: Path, monkeypatch):
    # Simulates what a look-ahead bug would produce: the return paired with
    # each ticker is the *same* regardless of which session it gets aligned
    # to (day 1 and day 2 bars are identical per ticker), so the sentiment/
    # return correlation survives shuffling untouched - exactly the failure
    # mode NEXT_STEPS.md's "Done when" section names, and the reason this
    # gate exists.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    tickers_returns = [
        ("PCJEWELLER.NS", 105.0),
        ("FORTIS.NS", 106.0),
        ("BSE.NS", 104.0),
        ("POLICYBZR.NS", 107.0),
        ("NSE.BO", 95.0),
        ("SUZLON.NS", 94.0),
        ("HDFCBANK.NS", 96.0),
        ("GESHIP.NS", 93.0),
    ]
    for ticker, close in tickers_returns:
        save_fixture(
            ticker,
            [
                Bar(date=DAY_1, open=100.0, close=close),
                Bar(date=DAY_2, open=100.0, close=close),  # identical to day 1 - timing is irrelevant
            ],
        )

    headlines = _build_engineered_headlines()

    result = run_audit(headlines, n_shuffles=300, seed=42, live=False)

    assert result.real_r > 0.6
    # Every shuffle reproduces essentially the same correlation, because the
    # return a ticker gets paired with never actually depended on which
    # session (and therefore which timestamp) it was aligned to.
    assert result.shuffled_mean > 0.5
    assert result.leak_suspected
