from datetime import date, datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    ResolvedHeadline,
    correlate_rows,
    pair_with_timestamps,
    permutation_p_value,
    resolve_headlines,
    run,
    run_audit,
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


# --- the seam the shuffle test relies on: a timestamp independent of the
# headline it's paired with still drives alignment -----------------------


def test_pair_with_timestamps_uses_the_given_timestamp_not_the_headlines_own():
    bars = [
        Bar(date=date(2026, 9, 28), open=100.0, close=105.0),
        Bar(date=date(2026, 9, 29), open=105.0, close=100.0),
    ]
    rh = ResolvedHeadline(
        title="x",
        ticker="X.NS",
        compound=0.5,
        published_at=datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),  # pre-open on the 29th
        bars=bars,
    )

    real_rows = pair_with_timestamps([rh], [rh.published_at])
    assert real_rows[0]["session_date"] == date(2026, 9, 29)
    assert real_rows[0]["contemporaneous_return"] == (100.0 - 105.0) / 105.0

    other_ts = datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)  # pre-open on the 28th instead
    other_rows = pair_with_timestamps([rh], [other_ts])
    assert other_rows[0]["session_date"] == date(2026, 9, 28)
    assert other_rows[0]["contemporaneous_return"] == (105.0 - 100.0) / 100.0


def test_pair_with_timestamps_drops_a_headline_with_no_bar_on_its_aligned_session():
    bars = [Bar(date=date(2026, 9, 28), open=100.0, close=105.0)]
    rh = ResolvedHeadline(
        title="x", ticker="X.NS", compound=0.5,
        published_at=datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
        bars=bars,
    )
    assert pair_with_timestamps([rh], [rh.published_at]) == []


def test_correlate_rows_degrades_gracefully_below_two_points():
    rows = [{"compound": 0.5, "contemporaneous_return": 0.01, "lagged_return": None}]
    stat = correlate_rows(rows)
    assert stat.contemporaneous_r is None
    assert stat.lagged_r is None
    assert stat.n_contemporaneous == 1
    assert stat.n_lagged == 0


def test_permutation_p_value_is_small_when_real_beats_every_shuffle():
    # 0 of 4 shuffles are as extreme as the real statistic -> (0+1)/(4+1)
    assert permutation_p_value(0.9, [0.1, 0.2, -0.1, 0.3]) == 1 / 5


def test_permutation_p_value_is_one_when_real_is_the_least_extreme():
    # every shuffle has |r| >= |real_r|=0 -> (3+1)/(3+1)
    assert permutation_p_value(0.0, [0.5, -0.4, 0.3]) == 1.0


def test_permutation_p_value_rejects_empty_shuffles():
    try:
        permutation_p_value(0.5, [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


# --- resolve_headlines: same resolution behaviour sentiment.correlate uses ---


def test_resolve_headlines_pairs_resolved_headlines_with_tickers(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=date(2026, 9, 28), open=100.0, close=105.0)])

    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            ),
            make_headline(
                "Cyient among 4 stocks showing White Marubozu Pattern",
                "2",
                datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc),
            ),
        ],
        in_path,
    )

    resolved, unresolved = resolve_headlines(in_path)

    assert unresolved == []
    assert len(resolved) == 1
    assert resolved[0].ticker == "INFY.NS"
    assert resolved[0].bars[0].close == 105.0


def test_resolve_headlines_reports_company_with_no_resolvable_ticker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "LTIMindtree Share Price Highlights: LTIMindtree Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            )
        ],
        in_path,
    )

    resolved, unresolved = resolve_headlines(in_path)

    assert resolved == []
    assert unresolved == [
        ("LTIMindtree Share Price Highlights: LTIMindtree Stock Price History", "LTIMindtree")
    ]


# --- the actual audit: a genuine, timing-dependent signal must not survive
# the shuffle; whatever the real (correctly-aligned) correlation reports is
# only convincing if label-shuffling it actually looks different ----------


def test_run_audit_shows_a_synthetic_timing_dependent_signal_does_not_survive_shuffling(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    # Six companies, alternating strongly positive/negative VADER sentiment
    # (checked by hand: all |compound| > 0.85, consistent sign per company).
    headlines = [
        ("Infosys Share Price Highlights: outstanding profit surge, excellent and wonderful results", +1),
        ("Wipro Share Price Highlights: disastrous losses, horrible and terrible crash", -1),
        ("HUL Share Price Highlights: fantastic record growth, great and superb earnings beat", +1),
        ("Tata Steel Share Price Highlights: awful collapse, dreadful and disgusting decline", -1),
        ("Sun Pharma Share Price Highlights: amazing breakthrough, brilliant and excellent rally", +1),
        ("Grasim Inds Share Price Highlights: terrible failure, miserable and horrific plunge", -1),
    ]
    tickers = ["INFY.NS", "WIPRO.NS", "HINDUNILVR.NS", "TATASTEEL.NS", "SUNPHARMA.NS", "GRASIM.NS"]
    # Six distinct weekdays, one per headline - every headline is pre-open
    # (02:00 UTC = 07:30 IST) on its *own* day, so shuffling timestamps
    # across headlines genuinely changes which session_date each one lands
    # on, not just a no-op permutation of identical values.
    own_days = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)]

    # Each ticker's bars: on its *own* day the return matches its sentiment
    # sign (the genuine, timing-dependent relationship). On any *other* day
    # the return is a small fixed value that depends only on which day it
    # is, never on this ticker's own sentiment - a headline misrouted there
    # by shuffling carries no relationship to its own compound score at all,
    # rather than a uniform sign-flip (which would keep |r| unchanged and
    # make the permutation test blind to the difference).
    off_day_return = [0.020, -0.015, 0.008, -0.022, 0.011, -0.004]
    for i, (ticker, (_, sign), own_day) in enumerate(zip(tickers, headlines, own_days)):
        bars = []
        for k, day in enumerate(own_days):
            move = sign * 0.05 if k == i else off_day_return[k]
            bars.append(Bar(date=day, open=100.0, close=100.0 * (1 + move)))
        save_fixture(ticker, bars)

    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(title, str(i), datetime(own_day.year, own_day.month, own_day.day, 2, 0, 0, tzinfo=timezone.utc))
            for i, ((title, _), own_day) in enumerate(zip(headlines, own_days))
        ],
        in_path,
    )

    resolved, unresolved = resolve_headlines(in_path)
    assert unresolved == []
    assert len(resolved) == 6

    result = run_audit(resolved, n_shuffles=300, seed=0)

    # The real, correctly-aligned pipeline finds a strong signal by
    # construction (sentiment sign and same-day return sign always agree).
    assert result.real.contemporaneous_r > 0.9

    # Shuffling the timestamps routes most headlines to a day where the
    # relationship is reversed, so the shuffled null clusters well away
    # from the real statistic - the real r is an outlier against it.
    assert result.contemporaneous_p is not None
    assert result.contemporaneous_p < 0.05
    mean_shuffled = sum(result.shuffled_contemporaneous) / len(result.shuffled_contemporaneous)
    assert mean_shuffled < result.real.contemporaneous_r - 0.5


def test_run_is_deterministic_for_a_fixed_seed(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=date(2026, 9, 29), open=105.0, close=103.0),
        ],
    )
    save_fixture(
        "HINDUNILVR.NS",
        [
            Bar(date=date(2026, 9, 28), open=200.0, close=198.0),
            Bar(date=date(2026, 9, 29), open=198.0, close=201.0),
        ],
    )
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            ),
            make_headline(
                "HUL Share Price Highlights: HUL Stock Price History",
                "2",
                datetime(2026, 9, 28, 2, 30, 0, tzinfo=timezone.utc),
            ),
        ],
        in_path,
    )

    exit_code_1 = run(in_path, live=False, n_shuffles=20, seed=5)
    out_1 = capsys.readouterr().out
    exit_code_2 = run(in_path, live=False, n_shuffles=20, seed=5)
    out_2 = capsys.readouterr().out

    assert exit_code_1 == 0
    assert exit_code_2 == 0
    assert out_1 == out_2


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, live=False, n_shuffles=20, seed=0)
    assert exit_code == 1


def test_run_reports_failure_with_fewer_than_two_resolved_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=date(2026, 9, 28), open=100.0, close=105.0)])
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            )
        ],
        in_path,
    )
    exit_code = run(in_path, live=False, n_shuffles=20, seed=0)
    assert exit_code == 1


def test_run_against_committed_fixture_matches_days_5_and_6_real_statistics(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=50, seed=0)

    assert exit_code == 0
    out = capsys.readouterr().out
    # Day 5/6 Findings recorded these exact real statistics - the audit
    # must reproduce them, not a different pairing.
    assert "real r=-0.185 (n=23)" in out
    assert "real r=+0.000 (n=15)" in out
